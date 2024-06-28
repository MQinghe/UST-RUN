import argparse
import logging
import os
import random
import shutil
import sys
import time
from typing import Iterable

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from networks.unet_model import UNet
# from networks.unet import UNet
from networks.wrn import build_WideResNet
from dataloaders.dataloader import FundusSegmentation, ProstateSegmentation, MNMSSegmentation
import dataloaders.custom_transforms as tr
from utils import losses, metrics, ramps, util
from torch.cuda.amp import autocast, GradScaler
import contextlib
import matplotlib.pyplot as plt 

from torch.optim.lr_scheduler import LambdaLR
import math

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='prostate', choices=['fundus', 'prostate'])
parser.add_argument("--save_name", type=str, default="debug", help="experiment_name")
parser.add_argument("--overwrite", action='store_true')
parser.add_argument("--model", type=str, default="unet", help="model_name")
parser.add_argument("--max_iterations", type=int, default=60000, help="maximum epoch number to train")
parser.add_argument('--num_eval_iter', type=int, default=500)
parser.add_argument("--deterministic", type=int, default=1, help="whether use deterministic training")
parser.add_argument("--base_lr", type=float, default=0.03, help="segmentation network learning rate")
parser.add_argument("--seed", type=int, default=1337, help="random seed")
parser.add_argument("--gpu", type=str, default='0')
parser.add_argument('--load',action='store_true')
parser.add_argument('--eval',action='store_true')
parser.add_argument('--load_path',type=str,default='../model/lb1_ratio0.2/iter_6000.pth')
parser.add_argument("--threshold", type=float, default=0.95, help="confidence threshold for using pseudo-labels",)

parser.add_argument('--amp', type=int, default=1, help='use mixed precision training or not')

parser.add_argument("--label_bs", type=int, default=4, help="labeled_batch_size per gpu")
parser.add_argument("--unlabel_bs", type=int, default=4)
parser.add_argument("--test_bs", type=int, default=1)
parser.add_argument('--domain_num', type=int, default=6)
parser.add_argument('--lb_domain', type=int, default=1)
parser.add_argument('--lb_num', type=int, default=40)
# costs
parser.add_argument("--ema_decay", type=float, default=0.99, help="ema_decay")
parser.add_argument("--consistency_type", type=str, default="mse", help="consistency_type")
parser.add_argument("--consistency", type=float, default=1.0, help="consistency")
parser.add_argument("--consistency_rampup", type=float, default=200.0, help="consistency_rampup")

parser.add_argument('--depth', type=int, default=28)
parser.add_argument('--widen_factor', type=int, default=2)
parser.add_argument('--leaky_slope', type=float, default=0.1)
parser.add_argument('--bn_momentum', type=float, default=0.1)
parser.add_argument('--dropout', type=float, default=0.0)

parser.add_argument("--cutmix_prob", default=1.0, type=float)
parser.add_argument('--save_img',action='store_true')
parser.add_argument("--increase", default=1.0005, type=float)
parser.add_argument("--queue_len", default=20, type=int)
args = parser.parse_args()


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # teacher network: ema_model
    # student network: model
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def cycle(iterable: Iterable):
    """Make an iterator returning elements from the iterable.

    .. note::
        **DO NOT** use `itertools.cycle` on `DataLoader(shuffle=True)`.\n
        Because `itertools.cycle` saves a copy of each element, batches are shuffled only at the first epoch. \n
        See https://docs.python.org/3/library/itertools.html#itertools.cycle for more details.
    """
    while True:
        for x in iterable:
            yield x

def get_SGD(net, name='SGD', lr=0.1, momentum=0.9, \
                  weight_decay=5e-4, nesterov=True, bn_wd_skip=True):
    '''
    return optimizer (name) in torch.optim.
    If bn_wd_skip, the optimizer does not apply
    weight decay regularization on parameters in batch normalization.
    '''
    optim = getattr(torch.optim, name)
    
    decay = []
    no_decay = []
    for name, param in net.named_parameters():
        if ('bn' in name) and bn_wd_skip:
            no_decay.append(param)
        else:
            decay.append(param)
    
    per_param_args = [{'params': decay},
                      {'params': no_decay, 'weight_decay': 0.0}]
    
    optimizer = optim(per_param_args, lr=lr,
                    momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    return optimizer
        
        
def get_cosine_schedule_with_warmup(optimizer,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    num_warmup_steps=0,
                                    last_epoch=-1):
    '''
    Get cosine scheduler (LambdaLR).
    if warmup is needed, set num_warmup_steps (int) > 0.
    '''
    
    def _lr_lambda(current_step):
        '''
        _lr_lambda returns a multiplicative factor given an interger parameter epochs.
        Decaying criteria: last_epoch
        '''
        
        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr
    
    return LambdaLR(optimizer, _lr_lambda, last_epoch)

if args.dataset == 'fundus':
    part = ['cup', 'disc']
    dataset = FundusSegmentation
elif args.dataset == 'prostate':
    part = ['base'] 
    dataset = ProstateSegmentation
n_part = len(part)
dice_calcu = {'fundus':metrics.dice_coeff_2label, 'prostate':metrics.dice_coeff}

@torch.no_grad()
def test(args, model, test_dataloader, epoch, writer, ema=True):
    model.eval()
    model_name = 'ema' if ema else 'stu'
    val_loss = 0.0
    val_dice = [0.0] * n_part
    domain_metrics = []
    domain_num = len(test_dataloader)
    if args.dataset == 'fundus':
        ce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        softmax, sigmoid, multi = False, True, True
    elif args.dataset == 'prostate':
        ce_loss = CrossEntropyLoss(reduction='none')
        softmax, sigmoid, multi = True, False, False
    dice_loss = losses.DiceLossWithMask(2)
    for i in range(domain_num):
        cur_dataloader = test_dataloader[i]
        dc = -1
        domain_val_loss = 0.0
        domain_val_dice = [0.0] * n_part
        for batch_num,sample in enumerate(cur_dataloader):
            dc = sample['dc'][0].item()
            data = sample['image'].cuda()
            mask = sample['label'].cuda()
            if args.dataset == 'fundus':
                cup_mask = mask.eq(0).float()
                disc_mask = mask.le(128).float()
                mask = torch.cat((cup_mask.unsqueeze(1), disc_mask.unsqueeze(1)),dim=1)
            elif args.dataset == 'prostate':
                mask = mask.eq(0).long()
            output = model(data)
            # loss_seg = torch.nn.BCEWithLogitsLoss()(output, mask)
            loss_seg = ce_loss(output, mask).mean() + \
                        dice_loss(output, mask.unsqueeze(1), softmax=softmax, sigmoid=sigmoid, multi=multi)

            if args.dataset == 'fundus':
                dice = dice_calcu[args.dataset](np.asarray(torch.sigmoid(output.cpu()))>=0.5,mask.clone().cpu())
            elif args.dataset == 'prostate':
                dice = dice_calcu[args.dataset](np.asarray(torch.max(torch.softmax(output.cpu(),dim=1), dim=1)[1]),mask.clone().cpu())
            
            domain_val_loss += loss_seg.item()
            for i in range(len(domain_val_dice)):
                domain_val_dice[i] += dice[i]
            if epoch % 10 == 0 and args.save_img and False:
                if args.dataset == 'fundus':
                    grid_image = make_grid([make_grid(data[0, ...].clone().cpu().data, 1, normalize=True), 
                            mask[0, 0, ...].clone().unsqueeze(0).repeat(3,1,1).cpu().data, 
                            torch.sigmoid(output)[0, 0, ...].clone().unsqueeze(0).repeat(3,1,1).ge(0.5).float().cpu().data, 
                            mask[0, 1, ...].clone().unsqueeze(0).repeat(3,1,1).cpu().data, 
                            torch.sigmoid(output)[0, 1, ...].clone().unsqueeze(0).repeat(3,1,1).ge(0.5).float().cpu().data],5,padding=2, pad_value=1)
                elif args.dataset == 'prostate':
                    grid_image = make_grid([make_grid(data[0, ...].clone().cpu().data, 1, normalize=True), 
                            mask[0, ...].clone().unsqueeze(0).repeat(3,1,1).cpu().data, 
                            torch.max(torch.softmax(output,dim=1), dim=1)[1][0, ...].clone().unsqueeze(0).repeat(3,1,1).float().cpu().data],
                            3,padding=2, pad_value=1)

                writer.add_image('{}_val/domain{}/{}'.format(model_name, dc,batch_num), grid_image, epoch)
        
        domain_val_loss /= len(cur_dataloader)
        val_loss += domain_val_loss
        writer.add_scalar('{}_val/domain{}/loss'.format(model_name, dc), domain_val_loss, epoch)
        for i in range(len(domain_val_dice)):
            domain_val_dice[i] /= len(cur_dataloader)
            val_dice[i] += domain_val_dice[i]
        for n, p in enumerate(part):
            writer.add_scalar('{}_val/domain{}/val_{}_dice'.format(model_name, dc, p), domain_val_dice[n], epoch)
        text = 'domain%d epoch %d : loss : %f' % (dc, epoch, domain_val_loss)
        for n, p in enumerate(part):
            text += ' val_%s_dice: %f' % (p, domain_val_dice[n])
            if n != n_part-1:
                text += ','
        logging.info(text)
        
    model.train()
    val_loss /= domain_num
    writer.add_scalar('{}_val/loss'.format(model_name), val_loss, epoch)
    for i in range(len(val_dice)):
        val_dice[i] /= domain_num
    for n, p in enumerate(part):
        writer.add_scalar('{}_val/val_{}_dice'.format(model_name, p), val_dice[n], epoch)
    text = 'epoch %d : loss : %f' % (epoch, val_loss)
    for n, p in enumerate(part):
        text += ' val_%s_dice: %f' % (p, val_dice[n])
        if n != n_part-1:
            text += ','
    logging.info(text)
    return val_dice
    
def entropy_loss(logits: torch.Tensor):
    return - (logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(dim=1).mean()

def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size, img_size).cuda()
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask

def obtain_all_cover_box(region):
    loc=torch.nonzero(region).cpu().numpy()
    if len(loc) == 0:
        return obtain_cutmix_box(region.shape[0], p=1.0)
    mask = torch.zeros_like(region).cuda()
    y1, y2, x1, x2 = loc[0,0], loc[-1,0], min(loc[:,1]), max(loc[:,1])

    mask[y1:y2+1, x1:x2+1] = 1

    return mask

def train(args, snapshot_path):
    writer = SummaryWriter(snapshot_path + '/log')
    base_lr = args.base_lr

    if args.dataset == 'fundus':
        num_channels = 3
        patch_size = 256
        num_classes = 2
        args.label_bs = 4
        args.unlabel_bs = 4
        min_v, max_v = 0.5, 1.5
        fillcolor = 255
        args.max_iterations = 30000
        if args.domain_num >=4:
            args.domain_num = 4
    elif args.dataset == 'prostate':
        num_channels = 1
        patch_size = 384
        num_classes = 2
        args.label_bs = 4
        args.unlabel_bs = 4
        min_v, max_v = 0.1, 2
        fillcolor = 255
        args.max_iterations = 60000
        if args.domain_num >= 6:
            args.domain_num = 6

    max_iterations = args.max_iterations
    weak = transforms.Compose([tr.RandomScaleCrop(patch_size),
            tr.RandomScaleRotate(fillcolor=fillcolor),
            tr.RandomHorizontalFlip(),
            tr.elastic_transform(),
            ])
    
    strong = transforms.Compose([
            tr.Brightness(min_v, max_v),
            tr.Contrast(min_v, max_v),
            tr.GaussianBlur(kernel_size=int(0.1 * patch_size), num_channels=num_channels),
    ])

    normal_toTensor = transforms.Compose([
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    domain_num = args.domain_num
    domain = list(range(1,domain_num+1))
    if args.dataset == 'fundus':
        domain_len = [50, 99, 320, 320]
    elif args.dataset == 'prostate':
        domain_len = [225, 305, 136, 373, 338, 133]
    lb_domain = args.lb_domain
    data_num = domain_len[lb_domain-1]
    lb_num = args.lb_num
    lb_idxs = list(range(lb_num))
    unlabeled_idxs = list(range(lb_num, data_num))
    test_dataset = []
    test_dataloader = []
    lb_dataset = dataset(base_dir=train_data_path, phase='train', splitid=lb_domain, domain=[lb_domain], 
                                                selected_idxs = lb_idxs, weak_transform=weak,normal_toTensor=normal_toTensor)
    ulb_dataset = dataset(base_dir=train_data_path, phase='train', splitid=lb_domain, domain=domain, 
                                                selected_idxs=unlabeled_idxs, weak_transform=weak, strong_tranform=strong,normal_toTensor=normal_toTensor)
    for i in range(1, domain_num+1):
        cur_dataset = dataset(base_dir=train_data_path, phase='test', splitid=-1, domain=[i], normal_toTensor=normal_toTensor)
        test_dataset.append(cur_dataset)
    if not args.eval:
        lb_dataloader = cycle(DataLoader(lb_dataset, batch_size = args.label_bs, shuffle=True, num_workers=2, pin_memory=True, drop_last=True))
        ulb_dataloader = cycle(DataLoader(ulb_dataset, batch_size = args.unlabel_bs, shuffle=True, num_workers=2, pin_memory=True, drop_last=True))
    for i in range(0,domain_num):
        cur_dataloader = DataLoader(test_dataset[i], batch_size = args.test_bs, shuffle=False, num_workers=0, pin_memory=True)
        test_dataloader.append(cur_dataloader)

    def create_model(ema=False):
        # Network definition
        if args.model == 'unet':
            model = UNet(n_channels = num_channels, n_classes = num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model.cuda()

    model = create_model()
    ema_model = create_model(ema=True)

    iter_num = 0
    start_epoch = 0

    # instantiate optimizers
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    # set to train
    if args.dataset == 'fundus':
        ce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        softmax, sigmoid, multi = False, True, True
    elif args.dataset == 'prostate':
        ce_loss = CrossEntropyLoss(reduction='none')
        softmax, sigmoid, multi = True, False, False
    dice_loss = losses.DiceLossWithMask(num_classes)

    logging.info("{} iterations per epoch".format(args.num_eval_iter))

    max_epoch = max_iterations // args.num_eval_iter
    best_dice = [0.0] * n_part
    best_dice_iter = [-1] * n_part
    best_avg_dice = 0.0
    best_avg_dice_iter = -1
    dice_of_best_avg = [0.0] * n_part
    stu_best_dice = [0.0] * n_part
    stu_best_dice_iter = [-1] *n_part
    stu_best_avg_dice = 0.0
    stu_best_avg_dice_iter = -1
    stu_dice_of_best_avg = [0.0] * n_part

    iter_num = int(iter_num)

    # iterator = tqdm(range(start_epoch, max_epoch), ncols=70)
    threshold = args.threshold

    if args.load:
        path_str = '../model/{}/{}/checkpoint.pth'.format(args.dataset,args.save_name)
        start_epoch, ema_model, model, optimizer, best_avg_dice, best_avg_dice_iter, stu_best_avg_dice, stu_best_avg_dice_iter = util.load_osmancheckpoint(
            path_str, ema_model, model, optimizer
        )
        iter_num = start_epoch*args.num_eval_iter
        logging.info('Models restored from epoch {}'.format(start_epoch))

    if args.eval:
        # for i in range(1,5):
        #     model.load_state_dict(torch.load('../model/lb{}_r0.2_fixmatch_th0.9/unet_disc_dice_best_model.pth'.format(i)))
        #     test(args, model,test_dataloader,i,writer)
        model.load_state_dict(torch.load('../model/prostate/pu_0.9probfda_lb{}_r0.2_th0.9_v2/unet_dice_best_model.pth'.format(args.lb_domain)))
        test(args, model,test_dataloader,args.lb_domain,writer)
        exit()

    scaler = GradScaler()
    amp_cm = autocast if args.amp else contextlib.nullcontext
    
    simple_ulb = None
    cor_pl = None
    cor_gt = None
    cor_hardness = []
    cor_dc = None
    cor_mask = None
    max_len = args.queue_len
    choice_th = 0.1
    from utils.util import AverageMeter as Avg
    for epoch_num in range(start_epoch, max_epoch):
        model.train()
        ema_model.train()
        p_bar = tqdm(range(args.num_eval_iter))
        p_bar.set_description(f'No. {epoch_num+1}')
        avg_hardness = Avg()
        avg_dice = [Avg() for i in range(n_part)]
        other_ulb_avg_dice = [Avg() for i in range(n_part)]
        all_ulb_avg_dice = [Avg() for i in range(n_part)]
        lq_avg_dice = [Avg() for i in range(n_part)]
        dc_record = [0] * domain_num
        simple_ulb_name = {}
        lq_u, lq_pl, lq_mask = None, None, None
        for i_batch in range(1, args.num_eval_iter+1):
            lb_sample = next(lb_dataloader)
            ulb_sample = next(ulb_dataloader)
            lb_x_w, lb_y = lb_sample['image'], lb_sample['label']
            ulb_x_w, ulb_x_s, ulb_y = ulb_sample['image'], ulb_sample['strong_aug'], ulb_sample['label']
            lb_dc, ulb_dc = lb_sample['dc'].cuda(), ulb_sample['dc'].cuda()
            ulb_dc_type = ulb_dc.clone()
            inlier = ulb_dc == lb_dc[0]
            outlier = ulb_dc != lb_dc[0]
            lb_name = lb_sample['img_name']
            ulb_name = ulb_sample['img_name']

            lb_x_w, lb_y, ulb_x_w, ulb_x_s, ulb_y = lb_x_w.cuda(), lb_y.cuda(), ulb_x_w.cuda(), ulb_x_s.cuda(), ulb_y.cuda()
            if args.dataset == 'fundus':
                lb_cup_label = lb_y.eq(0).float() # == 0
                lb_disc_label = lb_y.le(128).float()  # <= 128
                lb_mask = torch.cat((lb_cup_label.unsqueeze(1), lb_disc_label.unsqueeze(1)),dim=1)
                ulb_cup_label = ulb_y.eq(0).float()
                ulb_disc_label = ulb_y.le(128).float()
                ulb_mask = torch.cat((ulb_cup_label.unsqueeze(1), ulb_disc_label.unsqueeze(1)),dim=1)
                lb_mask_shape = [len(lb_x_w), num_classes, patch_size, patch_size]
                ulb_mask_shape = [len(ulb_x_w), num_classes, patch_size, patch_size]
            elif args.dataset == 'prostate':
                lb_mask = lb_y.eq(0).long()
                ulb_mask = ulb_y.eq(0).long()
                lb_mask_shape = [len(lb_x_w), 1, patch_size, patch_size]
                ulb_mask_shape = [len(ulb_x_w), 1, patch_size, patch_size]

            with amp_cm():

                if simple_ulb is None or len(simple_ulb) == 0:
                    cut_img = lb_x_w.clone()
                    cut_label = lb_mask.clone()
                    cut_mask = torch.ones(lb_mask_shape).cuda()
                    choice = np.random.randint(0, len(lb_x_w), len(ulb_x_s))
                else:
                    cut_img = torch.cat((lb_x_w.clone(), simple_ulb), dim=0)
                    cut_label = torch.cat((lb_mask.clone(), cor_pl), dim=0)
                    cut_mask = torch.cat((torch.ones(lb_mask_shape).cuda(), cor_mask), dim=0)
                    choice_in_simple_num = min(int(len(ulb_x_s)*0.5), len(simple_ulb))
                    choice_in_lb_num = len(ulb_x_s) - choice_in_simple_num
                    choice_in_lb = np.random.randint(0,len(lb_x_w), choice_in_lb_num)
                    choice_in_simple = np.random.randint(len(lb_x_w), len(lb_x_w)+len(simple_ulb), choice_in_simple_num)
                    choice = np.random.permutation(np.concatenate((choice_in_lb, choice_in_simple)))
                label_box = torch.stack([obtain_cutmix_box(img_size=patch_size, p=args.cutmix_prob) for i in range(len(ulb_x_s))], dim=0)
                img_box = label_box.unsqueeze(1)
                if args.dataset == 'fundus':
                    label_box = label_box.unsqueeze(1)
                mix_img = cut_img[choice]
                ulb_x_s_ul = ulb_x_s * (1-img_box) + mix_img * img_box
                # outputs for model
                logits_lb_x_w = model(lb_x_w)
                logits_ulb_x_w = ema_model(ulb_x_w)
                logits_ulb_x_s_ul = model(ulb_x_s_ul)
                stu_logits_ulb_x_w = model(ulb_x_w).detach()
                
                if args.dataset == 'fundus':
                    prob = logits_ulb_x_w.sigmoid()
                    pseudo_label = prob.ge(0.5).float()
                    mask = prob.ge(threshold).float() + prob.le(1-threshold).float()
                    stu_prob_ulb_x_w = stu_logits_ulb_x_w.sigmoid()
                    stu_pseudo_label = stu_prob_ulb_x_w.ge(0.5).float()
                elif args.dataset == 'prostate':
                    prob_ulb_x_w = torch.softmax(logits_ulb_x_w, dim=1)
                    prob, pseudo_label = torch.max(prob_ulb_x_w, dim=1)
                    mask = (prob > threshold).unsqueeze(1).float()
                    stu_prob_ulb_x_w = torch.softmax(stu_logits_ulb_x_w, dim=1)
                    _, stu_pseudo_label = torch.max(stu_prob_ulb_x_w, dim=1)

                mask_ul = mask.clone()
                
                stu_tea_dice = dice_calcu[args.dataset](np.asarray(stu_pseudo_label.clone().cpu()), pseudo_label.clone().cpu(), ret_arr=True)
                # print(stu_tea_dice)
                tmp_dice = stu_tea_dice[0]
                for i in range(1, n_part):
                    tmp_dice += stu_tea_dice[i]
                stu_tea_dice = tmp_dice/n_part
                hardness = 1-stu_tea_dice
                if epoch_num == 0:
                    for i in range(len(hardness)):
                        hardness[i] = 1
                # print(hardness)
                lq_idx, max_v = 0, -1
                for i in range(len(hardness)):
                    if hardness[i]>max_v:
                        max_v = hardness[i]
                        lq_idx = i
                        
                if lq_u is not None:
                    new_choice = np.random.randint(0, len(lb_x_w))
                    if args.dataset == 'fundus':
                        region = lq_pl[0,1].clone()
                        region[lq_pl[0,0].long()==1] = 1
                        region[lb_mask[new_choice,0].long()==1] = 1
                        region[lb_mask[new_choice,1].long()==1] = 1
                    elif args.dataset == 'prostate':
                        region = lq_pl[0].clone()
                        region[lb_mask[new_choice].long()>0] = 1
                    label_box_lq = obtain_all_cover_box(region=region).unsqueeze(0)
                    img_box_lq = label_box_lq.unsqueeze(1)
                    if args.dataset == 'fundus':
                        label_box_lq = label_box_lq.unsqueeze(1)
                    lq_s = lq_u * (1-img_box_lq) + lb_x_w[[new_choice]] * img_box_lq
                    pseudo_label_lq = (lq_pl * (1-label_box_lq) + lb_mask[[new_choice]] * label_box_lq).long()
                    if args.dataset == 'fundus':
                        pseudo_label_lq = pseudo_label_lq.float()
                    mask_lq = lq_mask.clone()
                    mask_lq[img_box_lq.expand(mask_lq.shape) == 1] = 1
                    logits_lq_s = model(lq_s)
                else:
                    print('first')
                    logits_ul_lq = None
                
                lq_dice = dice_calcu[args.dataset](np.asarray(pseudo_label[[lq_idx]].clone().cpu()), ulb_mask[[lq_idx]].clone().cpu())
                for i in range(n_part):
                    lq_avg_dice[i].update(lq_dice[i])
                    
                lq_u = ulb_x_w[[lq_idx]].clone()
                lq_pl = pseudo_label[[lq_idx]].clone()
                lq_mask = mask[[lq_idx]].clone()
                
                simple_ulb_idx = hardness < choice_th
                cur_simple_num = simple_ulb_idx.astype(int).sum()
                if simple_ulb is None or len(simple_ulb) == 0:
                    simple_ulb = ulb_x_w[simple_ulb_idx].clone()
                    cor_pl = pseudo_label[simple_ulb_idx].clone()
                    cor_gt = ulb_mask[simple_ulb_idx].clone()
                    cor_hardness = hardness[simple_ulb_idx].copy()
                    cor_dc = ulb_dc[simple_ulb_idx]
                    cor_mask = mask[simple_ulb_idx].clone()
                    if len(simple_ulb) > 0:
                        choice_th = min(choice_th, cor_hardness.max())
                else:
                    if cur_simple_num > 0:
                        if len(simple_ulb)+cur_simple_num > max_len:
                            newlen = max_len - cur_simple_num
                        else:
                            newlen = len(simple_ulb)
                        simple_ulb = torch.cat((ulb_x_w[simple_ulb_idx].clone(), simple_ulb[:newlen]),dim=0)
                        cor_pl = torch.cat((pseudo_label[simple_ulb_idx].clone(), cor_pl[:newlen]), dim=0)
                        cor_gt = torch.cat((ulb_mask[simple_ulb_idx].clone(), cor_gt[:newlen]), dim=0)
                        cor_dc = torch.cat((ulb_dc[simple_ulb_idx].clone(), cor_dc[:newlen]), dim=0)
                        cor_hardness = np.concatenate((hardness[simple_ulb_idx].copy(), cor_hardness[:newlen]))
                        cor_mask = torch.cat((mask[simple_ulb_idx].clone(), cor_mask[:newlen]), dim=0)
                        choice_th = min(choice_th, cor_hardness.max())
                    else:
                        choice_th = min(args.increase*choice_th, 0.1)
                    

                assert len(simple_ulb) == len(cor_pl) and len(cor_pl) == len(cor_gt) and len(cor_gt) == len(cor_hardness) and len(cor_hardness) == len(cor_dc) and len(cor_dc) == len(cor_mask)
                if cur_simple_num > 0:
                    cur_simple_ulb_dice = dice_calcu[args.dataset](np.asarray(pseudo_label[simple_ulb_idx].clone().cpu()), ulb_mask[simple_ulb_idx].clone().cpu())
                    for i in range(n_part):
                        avg_dice[i].update(cur_simple_ulb_dice[i])
                    avg_hardness.update(hardness[simple_ulb_idx].mean())
                    for idx, flag in enumerate(simple_ulb_idx):
                        if flag:
                            dc_record[ulb_dc[idx].item()-1] += 1
                            if ulb_name[idx] in simple_ulb_name:
                                simple_ulb_name[ulb_name[idx]] += 1
                            else:
                                simple_ulb_name[ulb_name[idx]] = 1

                other_ulb_idx = ~simple_ulb_idx
                cur_other_ulb_num = other_ulb_idx.astype(int).sum()
                if len(simple_ulb) > 0:
                    simple_ulb_dice = dice_calcu[args.dataset](np.asarray(cor_pl.clone().cpu()), cor_gt.clone().cpu())
                else:
                    simple_ulb_dice = [-1]*n_part
                
                if cur_other_ulb_num > 0:
                    other_ulb_dice = dice_calcu[args.dataset](np.asarray(pseudo_label[other_ulb_idx].clone().cpu()), ulb_mask[other_ulb_idx].clone().cpu())
                    for i in range(n_part):
                        other_ulb_avg_dice[i].update(other_ulb_dice[i])
                ulb_dice = dice_calcu[args.dataset](np.asarray(pseudo_label.clone().cpu()), ulb_mask.clone().cpu())
                for i in range(n_part):
                    all_ulb_avg_dice[i].update(ulb_dice[i])
                text = 'avg_hardness: {}'.format(avg_hardness.avg)
                for n, p in enumerate(part):
                    text += '  simple dice {} {}'.format(p, simple_ulb_dice[n])
                # print(text)
                for n, p in enumerate(part):
                    text = "avg_dice_{}: {} other_ulb_avg_dice_{}: {} all_ulb_avg_dice_{}: {}".format(p, avg_dice[n].avg, p, other_ulb_avg_dice[n].avg, p, all_ulb_avg_dice[n].avg)
                    
                sup_loss = ce_loss(logits_lb_x_w, lb_mask).mean() + \
                            dice_loss(logits_lb_x_w, lb_mask.unsqueeze(1), softmax=softmax, sigmoid=sigmoid, multi=multi)

                consistency_weight = get_current_consistency_weight(
                    iter_num // (args.max_iterations/args.consistency_rampup))

                mask_ul[img_box.expand(mask_ul.shape) == 1] = cut_mask[choice][img_box.expand(mask_ul.shape) == 1]
                pseudo_label_ul = (pseudo_label * (1-label_box) + cut_label[choice] * label_box).long()
                if args.dataset == 'fundus':
                    pseudo_label_ul = pseudo_label_ul.float()
                
                if logits_ul_lq is not None:
                    logits_ul_lq = torch.cat((logits_ulb_x_s_ul, logits_lq_s),dim=0)
                    pseudo_label_ul_lq = torch.cat((pseudo_label_ul, pseudo_label_lq),dim=0)
                    mask_ul_lq = torch.cat((mask_ul, mask_lq),dim=0)
                    unsup_loss = (ce_loss(logits_ul_lq, pseudo_label_ul_lq) * mask_ul_lq.squeeze(1)).mean() + \
                                    dice_loss(logits_ul_lq, pseudo_label_ul_lq.unsqueeze(1), mask=mask_ul_lq, softmax=softmax, sigmoid=sigmoid, multi=multi)
                else:
                    unsup_loss = (ce_loss(logits_ulb_x_s_ul, pseudo_label_ul) * mask_ul.squeeze(1)).mean() + \
                                    dice_loss(logits_ulb_x_s_ul, pseudo_label_ul.unsqueeze(1), mask=mask_ul, softmax=softmax, sigmoid=sigmoid, multi=multi)
                
                loss = sup_loss + consistency_weight * unsup_loss

            optimizer.zero_grad()

            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # update ema model
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            # update learning rate
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            for n, p in enumerate(part):
                text = 'train/ulb_{}_dice'.format(p)
                writer.add_scalar(text, ulb_dice[n], iter_num)
            writer.add_scalar('train/mask', mask.mean(), iter_num)
            writer.add_scalar('train/lr', lr_, iter_num)
            writer.add_scalar('train/loss', loss.item(), iter_num)
            writer.add_scalar('train/sup_loss', sup_loss.item(), iter_num)
            writer.add_scalar('train/unsup_loss', unsup_loss.item(), iter_num)
            # writer.add_scalar('train/unsup_lq_loss', unsup_lq_loss.item(), iter_num)
            writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
            if p_bar is not None:
                p_bar.update()

            if args.dataset == 'fundus':
                p_bar.set_description('iteration %d: loss:%.4f,sup_loss:%.4f,unsup_loss:%.4f,cons_w:%.4f,mask_ratio:%.4f,ulb_cd:%.4f,ulb_dd:%.4f' 
                                        % (iter_num, loss.item(), sup_loss.item(), unsup_loss.item(), consistency_weight, mask.mean(), ulb_dice[0], ulb_dice[1]))
            elif args.dataset == 'prostate':
                p_bar.set_description('iteration %d : loss:%.4f, sup_loss:%.4f, unsup_loss:%.4f, cons_w:%.4f, mask_ratio:%.4f, ulb_dice:%.4f' 
                                    % (iter_num, loss.item(), sup_loss.item(), unsup_loss.item(), consistency_weight, mask.mean(), ulb_dice[0]))
            if iter_num % args.num_eval_iter == 0:
                if iter_num % (args.num_eval_iter*5) == 0 and args.save_img:
                    if args.dataset == 'fundus':
                        lb_image = make_grid([make_grid(lb_x_w[0, ...].clone().cpu().data,1,normalize=True), 
                                                lb_mask[0,0,...].clone().unsqueeze(0).repeat(3,1,1).cpu().data, 
                                                logits_lb_x_w.sigmoid()[0, 0, ...].clone().unsqueeze(0).repeat(3,1,1).ge(0.5).float().cpu().data, 
                                                lb_mask[0, 1, ...].clone().unsqueeze(0).repeat(3,1,1).cpu().data, 
                                                logits_lb_x_w.sigmoid()[0, 1, ...].clone().unsqueeze(0).repeat(3,1,1).ge(0.5).float().cpu().data],5,padding=2, pad_value=1)
                        ulb_image = make_grid([make_grid(ulb_x_w[0, ...].clone().cpu().data,1,normalize=True), 
                                                ulb_mask[0,0,...].clone().unsqueeze(0).repeat(3,1,1).cpu().data, 
                                                ulb_mask[0, 1, ...].clone().unsqueeze(0).repeat(3,1,1).cpu().data, 
                                                make_grid(ulb_x_s_ul[0, ...].clone().cpu().data,1,normalize=True),
                                                pseudo_label[0, 0, ...].clone().unsqueeze(0).repeat(3,1,1).cpu().data, 
                                                pseudo_label[0, 1, ...].clone().unsqueeze(0).repeat(3,1,1).cpu().data,
                                                pseudo_label_ul[0, 0, ...].clone().unsqueeze(0).repeat(3,1,1).cpu().data,
                                                pseudo_label_ul[0, 1, ...].clone().unsqueeze(0).repeat(3,1,1).cpu().data],3,padding=2, pad_value=1)
                    elif args.dataset == 'prostate':
                        lb_image = make_grid([make_grid(lb_x_w[0, ...].clone().cpu().data,1,normalize=True), 
                                                lb_mask[0,...].clone().unsqueeze(0).repeat(3,1,1).cpu().data, 
                                                torch.max(torch.softmax(logits_lb_x_w, dim=1), dim=1)[1][0, ...].clone().unsqueeze(0).repeat(3,1,1).cpu().data],
                                                3,padding=2, pad_value=1)
                        ulb_image = make_grid([make_grid(ulb_x_w[0, ...].clone().cpu().data,1,normalize=True), 
                                                ulb_mask[0,...].clone().unsqueeze(0).repeat(3,1,1).cpu().data, 
                                                make_grid(ulb_x_s[0, ...].clone().cpu().data,1,normalize=True), 
                                                pseudo_label[0, ...].clone().unsqueeze(0).repeat(3,1,1).cpu().data, 
                                                make_grid(ulb_x_s_ul[0, ...].clone().cpu().data,1,normalize=True), 
                                                torch.max(torch.softmax(logits_ulb_x_s_ul, dim=1), dim=1)[1][0, ...].clone().unsqueeze(0).repeat(3,1,1).cpu().data],
                                                2,padding=2, pad_value=1)
                    writer.add_image("train/lb_sample", lb_image, iter_num)
                    writer.add_image("train/ulb_sample", ulb_image, iter_num)
                logging.info('iteration %d : loss : %f, sup_loss : %f, unsup_loss : %f, cons_w : %f, mask_ratio : %f' 
                                    % (iter_num, loss.item(), sup_loss.item(), unsup_loss.item(), consistency_weight, mask.mean()))
                text = ''
                for n, p in enumerate(part):
                    text = 'cur simple dice avg %s:%f' % (p, simple_ulb_dice[n])
                    logging.info(text)
                
                logging.info('tmp simple hardness avg:%f' % avg_hardness.avg)
                logging.info('choice threshold:%f' % choice_th)
                for i in range(len(dc_record)):
                    logging.info('tmp simple domain %d cnt: %d' % (i+1, dc_record[i]))

        for n, p in enumerate(part):
            text = 'epoch simple dice avg %s:%f' % (p, avg_dice[n].avg)
            logging.info(text)
        for n, p in enumerate(part):
            text = 'epoch other ulb dice avg %s:%f' % (p, other_ulb_avg_dice[n].avg)
            logging.info(text)
        for n, p in enumerate(part):
            text = 'epoch all ulb dice avg %s:%f' % (p, all_ulb_avg_dice[n].avg)
            logging.info(text)
        for n, p in enumerate(part):
            text = 'epoch lq ulb dice avg %s:%f' % (p, lq_avg_dice[n].avg)
            logging.info(text)
        logging.info('epoch simple hardness avg:%f' % avg_hardness.avg)
        logging.info('choice threshold:%f' % choice_th)
        simple_ulb_cnt = ""
        for i in simple_ulb_name:
            simple_ulb_cnt = simple_ulb_cnt + i + " " + str(simple_ulb_name[i]) + " "
        logging.info(simple_ulb_cnt)
        for i in range(len(dc_record)):
            logging.info('epoch simple domain %d cnt: %d' % (i+1, dc_record[i]))

        if p_bar is not None:
            p_bar.close()


        logging.info('test ema model')
        text = ''
        val_dice = test(args, ema_model, test_dataloader, epoch_num+1, writer)
        for n, p in enumerate(part):
            if val_dice[n] > best_dice[n]:
                best_dice[n] = val_dice[n]
                best_dice_iter[n] = iter_num
            text += 'val_%s_best_dice: %f at %d iter' % (p, best_dice[n], best_dice_iter[n])
            text += ', '
        if sum(val_dice) / len(val_dice) > best_avg_dice:
            best_avg_dice = sum(val_dice) / len(val_dice)
            best_avg_dice_iter = iter_num
            for n, p in enumerate(part):
                dice_of_best_avg[n] = val_dice[n]
        text += 'val_best_avg_dice: %f at %d iter' % (best_avg_dice, best_avg_dice_iter)
        if n_part > 1:
            for n, p in enumerate(part):
                text += ', %s_dice: %f' % (p, dice_of_best_avg[n])
        logging.info(text)
        logging.info('test stu model')
        stu_val_dice = test(args, model, test_dataloader, epoch_num+1, writer, ema=False)
        text = ''
        for n, p in enumerate(part):
            if stu_val_dice[n] > stu_best_dice[n]:
                stu_best_dice[n] = stu_val_dice[n]
                stu_best_dice_iter[n] = iter_num
            text += 'stu_val_%s_best_dice: %f at %d iter' % (p, stu_best_dice[n], stu_best_dice_iter[n])
            text += ', '
        if sum(stu_val_dice) / len(stu_val_dice) > stu_best_avg_dice:
            stu_best_avg_dice = sum(stu_val_dice) / len(stu_val_dice)
            stu_best_avg_dice_iter = iter_num
            for n, p in enumerate(part):
                stu_dice_of_best_avg[n] = stu_val_dice[n]
            save_text = "{}_avg_dice_best_model.pth".format(args.model)
            save_best = os.path.join(snapshot_path, save_text)
            logging.info('save cur best avg model to {}'.format(save_best))
            torch.save(model.state_dict(), save_best)
        text += 'val_best_avg_dice: %f at %d iter' % (stu_best_avg_dice, stu_best_avg_dice_iter)
        if n_part > 1:
            for n, p in enumerate(part):
                text += ', %s_dice: %f' % (p, stu_dice_of_best_avg[n])
        logging.info(text)
        text = 'checkpoint.pth'
        checkpoint_path = os.path.join(snapshot_path, text)
        util.save_osmancheckpoint(epoch_num+1, ema_model, model, optimizer, best_avg_dice, best_avg_dice_iter, stu_best_avg_dice, stu_best_avg_dice_iter, checkpoint_path)
        logging.info('save checkpoint to {}'.format(checkpoint_path))

        
    writer.close()


if __name__ == "__main__":
    snapshot_path = "../model/" + args.dataset + "/" + args.save_name + "/"
    if args.dataset == 'fundus':
        train_data_path='../../data/Fundus'
    elif args.dataset == 'prostate':
        train_data_path="../../data/ProstateSlice"
    elif args.dataset == 'MNMS':
        train_data_path="../../data/MNMS/mnms"

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    elif not args.overwrite:
        raise Exception('file {} is exist!'.format(snapshot_path))
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    cmd = " ".join(["python"] + sys.argv)
    logging.info(cmd)
    logging.info(str(args))

    train(args, snapshot_path)
