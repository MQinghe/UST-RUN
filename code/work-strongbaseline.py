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
from Fundus_dataloaders.fundus_dataloader import FundusSegmentation, ProstateSegmentation, MNMSSegmentation
import Fundus_dataloaders.custom_transforms as tr
from utils import losses, metrics, ramps, util
from torch.cuda.amp import autocast, GradScaler
import contextlib
import matplotlib.pyplot as plt 

from torch.optim.lr_scheduler import LambdaLR
import math
import scipy.misc

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='prostate', choices=['fundus', 'prostate', 'MNMS'])
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
parser.add_argument("--threshold", type=float, default=0.9, help="confidence threshold for using pseudo-labels",)

parser.add_argument('--amp', type=int, default=1, help='use mixed precision training or not')

parser.add_argument("--label_bs", type=int, default=2, help="labeled_batch_size per gpu")
parser.add_argument("--unlabel_bs", type=int, default=4)
parser.add_argument("--test_bs", type=int, default=4)
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

parser.add_argument("--beta", default=1.0, type=float)
parser.add_argument("--cutmix_prob", default=1.0, type=float)
parser.add_argument("--test_stu", default=True, action='store_true')

parser.add_argument("--increase", default=1.0005, type=float)
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
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

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

@torch.no_grad()
def test(args, model, test_dataloader, epoch, writer, ema=True):
    model.eval()
    model_name = 'ema' if ema else 'stu'
    val_loss = 0.0
    if args.dataset == 'fundus':
        val_cup_dice, val_disc_dice = 0.0, 0.0
    elif args.dataset == 'prostate':
        val_dice = 0.0
    elif args.dataset == 'MNMS':
        val_lv_dice, val_myo_dice, val_rv_dice = 0.0, 0.0, 0.0
    domain_metrics = []
    domain_num = len(test_dataloader)
    for i in range(domain_num):
        cur_dataloader = test_dataloader[i]
        dc = -1
        domain_val_loss = 0.0
        if args.dataset == 'fundus':
            domain_val_cup_dice, domain_val_disc_dice = 0.0, 0.0
        elif args.dataset == 'prostate':
            domain_val_dice = 0.0
        elif args.dataset == 'MNMS':
            domain_val_lv_dice, domain_val_myo_dice, domain_val_rv_dice = 0.0, 0.0, 0.0
        for batch_num,sample in enumerate(cur_dataloader):
            dc = sample['dc'][0].item()
            data = sample['image'].cuda()
            mask = sample['label'].cuda()
            if args.dataset == 'fundus':
                cup_mask = mask.eq(0).float()
                disc_mask = mask.le(128).float()
                mask = torch.cat((cup_mask, disc_mask),dim=1)
            elif args.dataset == 'prostate':
                mask = mask.eq(0).float()
            elif args.dataset == 'MNMS':
                lv_mask = mask[:,0,...].eq(255).float()
                myo_mask = mask[:,1,...].eq(255).float()
                rv_mask = mask[:,2,...].eq(255).float()
                back_mask = 1-(lv_mask+myo_mask+rv_mask)
                mask = torch.stack((lv_mask, myo_mask, rv_mask, back_mask), dim=1)
            output = model(data)
            if args.dataset == 'fundus' or args.dataset == 'prostate':
                loss_seg = torch.nn.BCEWithLogitsLoss()(output, mask)
            elif args.dataset == 'MNMS':
                loss_seg = F.cross_entropy(output, torch.argmax(mask, dim=1))

            if args.eval:
                for j in range(len(data)):
                    if args.dataset == 'fundus':
                        dice_cup, dice_disc = metrics.dice_coeff_2label(np.asarray(torch.sigmoid(output[j].cpu()))>=0.5, mask[j])
                        grid_image = make_grid([make_grid(data[j, ...].clone().cpu().data, 1, normalize=True), 
                                mask[j, 0, ...].clone().unsqueeze(0).repeat(3,1,1).cpu().data, 
                                torch.sigmoid(output)[j, 0, ...].clone().unsqueeze(0).repeat(3,1,1).ge(0.5).float().cpu().data, 
                                mask[j, 1, ...].clone().unsqueeze(0).repeat(3,1,1).cpu().data, 
                                torch.sigmoid(output)[j, 1, ...].clone().unsqueeze(0).repeat(3,1,1).ge(0.5).float().cpu().data],5,padding=2, pad_value=1)
                        if dice_cup<0.8 or dice_disc<0.8:
                            writer.add_image('lb_domain{}/bad/domain{}/{}_{}'.format(epoch,dc,round(dice_cup,4), round(dice_disc,4)), grid_image, 1)
                        else:
                            writer.add_image('lb_domain{}/good/domain{}/{}_{}'.format(epoch,dc,round(dice_cup,4), round(dice_disc,4)), grid_image, 1)
                    elif args.dataset == 'prostate':
                        dice = metrics.dice_coeff(np.asarray(torch.sigmoid(output[j].cpu()))>=0.5,mask[j].clone().cpu())
                        grid_image = make_grid([make_grid(data[j, ...].clone().cpu().data, 1, normalize=True), 
                                    mask[j, ...].clone().repeat(3,1,1).cpu().data, 
                                    torch.sigmoid(output)[j, 0, ...].clone().repeat(3,1,1).ge(0.5).float().cpu().data],3,padding=2, pad_value=1)
                        if dice<0.8:
                            writer.add_image('lb_domain{}/bad/domain{}/{}'.format(epoch,dc,round(dice,4)), grid_image, 1)
                        else:
                            writer.add_image('lb_domain{}/good/domain{}/{}'.format(epoch,dc,round(dice,4)), grid_image, 1)
                    elif args.dataset == 'MNMS':
                        test_prob = F.softmax(output[j], dim=0)
                        test_max_prob, test_max_idx = torch.max(test_prob, dim=0, keepdim=True)
                        test_lb = (test_prob == test_max_prob).float()
                        dice_lv, dice_myo, dice_rv = metrics.dice_coeff_3label(np.asarray(test_lb.cpu()), mask[j])
                        grid_image = make_grid([make_grid(data[j,:3, ...].clone().cpu().data, 1, normalize=True), 
                                mask[j,:3, ...].clone().cpu().data, 
                                test_lb[j,:3, ...].clone().cpu().data],3,padding=2, pad_value=1)
                        if dice_lv<0.8 or dice_myo<0.8 or dice_rv<0.8:
                            writer.add_image('lb_domain{}/bad/domain{}/{}_{}_{}'.format(epoch,dc,round(dice_lv,4), round(dice_myo,4), round(dice_rv,4)), grid_image, 1)
                        else:
                            writer.add_image('lb_domain{}/good/domain{}/{}_{}_{}'.format(epoch,dc,round(dice_lv,4), round(dice_myo,4), round(dice_rv, 4)), grid_image, 1)

            if args.dataset == 'fundus':
                dice_cup, dice_disc = metrics.dice_coeff_2label(np.asarray(torch.sigmoid(output.cpu()))>=0.5, mask)
            elif args.dataset == 'prostate':
                dice = metrics.dice_coeff(np.asarray(torch.sigmoid(output.cpu()))>=0.5,mask.clone().cpu())
            elif args.dataset == 'MNMS':
                test_prob = F.softmax(output, dim=1)
                test_max_prob, test_max_idx = torch.max(test_prob, dim=1, keepdim=True)
                test_lb = (test_prob == test_max_prob).float()
                dice_lv, dice_myo, dice_rv = metrics.dice_coeff_3label(np.asarray(test_lb.cpu()), mask)
            
            domain_val_loss += loss_seg.item()
            if args.dataset == 'fundus':
                domain_val_cup_dice += dice_cup
                domain_val_disc_dice += dice_disc
                grid_image = make_grid([make_grid(data[0, ...].clone().cpu().data, 1, normalize=True), 
                        mask[0, 0, ...].clone().unsqueeze(0).repeat(3,1,1).cpu().data, 
                        torch.sigmoid(output)[0, 0, ...].clone().unsqueeze(0).repeat(3,1,1).ge(0.5).float().cpu().data, 
                        mask[0, 1, ...].clone().unsqueeze(0).repeat(3,1,1).cpu().data, 
                        torch.sigmoid(output)[0, 1, ...].clone().unsqueeze(0).repeat(3,1,1).ge(0.5).float().cpu().data],5,padding=2, pad_value=1)
            elif args.dataset == 'prostate':
                domain_val_dice += dice
                grid_image = make_grid([make_grid(data[0, ...].clone().cpu().data, 1, normalize=True), 
                        mask[0, ...].clone().repeat(3,1,1).cpu().data, 
                        torch.sigmoid(output)[0, ...].clone().repeat(3,1,1).ge(0.5).float().cpu().data],5,padding=2, pad_value=1)
            elif args.dataset == 'MNMS':
                domain_val_lv_dice += dice_lv
                domain_val_myo_dice += dice_myo
                domain_val_rv_dice += dice_rv
                if epoch % 10 == 0:
                    grid_image = make_grid([make_grid(data[0, :3, ...].clone().cpu().data, 1, normalize=True), 
                            mask[0, :3, ...].clone().cpu().data, 
                            test_lb[0, :3, ...].clone().cpu().data],3,padding=2, pad_value=1)

            # if args.dataset == 'fundus' or args.dataset == 'prostate' or (args.dataset == 'MNMS' and epoch % 10 == 0):
            #     writer.add_image('{}_val/domain{}/{}'.format(model_name, dc,batch_num), grid_image, epoch)
        
        domain_val_loss /= len(cur_dataloader)
        val_loss += domain_val_loss
        writer.add_scalar('{}_val/domain{}/loss'.format(model_name, dc), domain_val_loss, epoch)
        if args.dataset == 'fundus':
            domain_val_cup_dice /= len(cur_dataloader)
            domain_val_disc_dice /= len(cur_dataloader)
            val_cup_dice += domain_val_cup_dice
            val_disc_dice += domain_val_disc_dice
            domain_metrics.append((domain_val_loss, domain_val_cup_dice, domain_val_disc_dice))
            writer.add_scalar('{}_val/domain{}/val_CUP_dice'.format(model_name, dc), domain_val_cup_dice, epoch)
            writer.add_scalar('{}_val/domain{}/val_DISC_dice'.format(model_name, dc), domain_val_disc_dice, epoch)
            logging.info('domain%d epoch %d : loss : %f val_cup_dice: %f, val_disc_dice:%f' %
                        (dc, epoch, domain_val_loss, domain_val_cup_dice,domain_val_disc_dice))
        elif args.dataset == 'prostate':
            domain_val_dice /= len(cur_dataloader)
            val_dice += domain_val_dice
            domain_metrics.append((domain_val_loss, domain_val_dice))
            writer.add_scalar('{}_val/domain{}/val_dice'.format(model_name, dc), domain_val_dice, epoch)
            logging.info('domain%d epoch %d : loss : %f val_dice: %f' %
                        (dc, epoch, domain_val_loss, domain_val_dice))
        elif args.dataset == 'MNMS':
            domain_val_lv_dice /= len(cur_dataloader)
            domain_val_myo_dice /= len(cur_dataloader)
            domain_val_rv_dice /= len(cur_dataloader)
            val_lv_dice += domain_val_lv_dice
            val_myo_dice += domain_val_myo_dice
            val_rv_dice += domain_val_rv_dice
            domain_metrics.append((domain_val_loss, domain_val_lv_dice, domain_val_myo_dice, domain_val_rv_dice))
            writer.add_scalar('{}_val/domain{}/val_LV_dice'.format(model_name, dc), domain_val_lv_dice, epoch)
            writer.add_scalar('{}_val/domain{}/val_MYO_dice'.format(model_name, dc), domain_val_myo_dice, epoch)
            writer.add_scalar('{}_val/domain{}/val_RV_dice'.format(model_name, dc), domain_val_rv_dice, epoch)
            logging.info('domain%d epoch %d : loss : %f val_lv_dice: %f, val_myo_dice:%f, val_rv_dice:%f' %
                        (dc, epoch, domain_val_loss, domain_val_lv_dice, domain_val_myo_dice, domain_val_rv_dice))
        
    model.train()
    val_loss /= domain_num
    writer.add_scalar('{}_val/loss'.format(model_name), val_loss, epoch)
    if args.dataset == 'fundus':
        val_cup_dice /= domain_num
        val_disc_dice /= domain_num
        writer.add_scalar('{}_val/val_CUP_dice'.format(model_name), val_cup_dice, epoch)
        writer.add_scalar('{}_val/val_DISC_dice'.format(model_name), val_disc_dice, epoch)
        logging.info('epoch %d : loss : %f val_cup_dice: %f, val_disc_dice:%f' %
                (epoch, val_loss, val_cup_dice,val_disc_dice))
        return val_cup_dice, val_disc_dice
    elif args.dataset == 'prostate':
        val_dice /= domain_num
        writer.add_scalar('{}_val/val_dice'.format(model_name), val_dice, epoch)
        logging.info('epoch %d : loss : %f val_dice: %f' %
                (epoch, val_loss, val_dice))
        return val_dice
    elif args.dataset == 'MNMS':
        val_lv_dice /= domain_num
        val_myo_dice /= domain_num
        val_rv_dice /= domain_num
        writer.add_scalar('{}_val/val_LV_dice'.format(model_name), val_lv_dice, epoch)
        writer.add_scalar('{}_val/val_MYO_dice'.format(model_name), val_myo_dice, epoch)
        writer.add_scalar('{}_val/val_RV_dice'.format(model_name), val_rv_dice, epoch)
        logging.info('epoch %d : loss : %f val_lv_dice: %f, val_myo_dice:%f, val_rv_dice:%f' %
                (epoch, val_loss, val_lv_dice, val_myo_dice, val_rv_dice))
        return val_lv_dice, val_myo_dice, val_rv_dice
    
def entropy_loss(logits: torch.Tensor):
    return - (logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(dim=1).mean()

def rand_bbox(size, lam):
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2

def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size, img_size)
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

def train(args, snapshot_path):
    writer = SummaryWriter(snapshot_path + '/log')
    base_lr = args.base_lr
    max_iterations = args.max_iterations

    if args.dataset == 'fundus':
        num_channels = 3
        patch_size = 256
        num_classes = 2
        args.label_bs = 2
        args.unlabel_bs = 8
        min_v, max_v = 0.5, 1.5
        fillcolor = 255
        if args.domain_num >=4:
            args.domain_num = 4
    elif args.dataset == 'prostate':
        num_channels = 1
        patch_size = 384
        num_classes = 1
        args.label_bs = 2
        args.unlabel_bs = 4
        min_v, max_v = 0.1, 2
        fillcolor = 255
        if args.domain_num >= 6:
            args.domain_num = 6
    elif args.dataset == 'MNMS':
        num_channels = 1
        patch_size = 224
        num_classes = 4
        args.label_bs = 2
        args.unlabel_bs = 8
        min_v, max_v = 0.1, 2
        fillcolor = 0
        if args.domain_num >= 4:
            args.domain_num = 4

    weak = transforms.Compose([tr.RandomScaleCrop(patch_size),
            # tr.RandomCrop(512),
            tr.RandomScaleRotate(fillcolor=fillcolor),
            # tr.RandomRotate(),
            tr.RandomHorizontalFlip(),
            # tr.RandomFlip(),
            tr.elastic_transform(),
            # tr.add_salt_pepper_noise(),
            # tr.adjust_light(),
            # tr.eraser(),
            # tr.Normalize_tf(),
            # tr.ToTensor()
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
        domain_len = [225, 305, 136, 468, 338, 133]
    elif args.dataset == 'MNMS':
        domain_len = [976, 1261, 505, 523]
    lb_domain = args.lb_domain
    data_num = domain_len[lb_domain-1]
    lb_num = args.lb_num
    lb_idxs = list(range(lb_num))
    unlabeled_idxs = list(range(lb_num, data_num))
    test_dataset = []
    test_dataloader = []
    if args.dataset == 'fundus':
        if not args.eval:
            lb_dataset = FundusSegmentation(base_dir=train_data_path, phase='train', splitid=lb_domain, domain=[lb_domain], 
                                                selected_idxs = lb_idxs, weak_transform=weak,normal_toTensor=normal_toTensor)
            ulb_dataset = FundusSegmentation(base_dir=train_data_path, phase='train', splitid=lb_domain, domain=domain, 
                                                selected_idxs=unlabeled_idxs, weak_transform=weak, strong_tranform=strong,normal_toTensor=normal_toTensor)
        for i in range(1,domain_num+1):
            cur_dataset = FundusSegmentation(base_dir=train_data_path, phase='test', splitid=-1, domain=[i], normal_toTensor=normal_toTensor)
            test_dataset.append(cur_dataset)
    elif args.dataset == 'prostate':
        # start = np.random.randint(0,data_num-lb_num)
        # logging.info('sample start from %d' % (start))
        # lb_idxs = list(range(start, start + lb_num))
        # total = list(range(data_num))
        # unlabeled_idxs = [x for x in total if x not in lb_idxs]
        if not args.eval:
            lb_dataset = ProstateSegmentation(base_dir=train_data_path, phase='train', splitid=lb_domain, domain=[lb_domain], 
                                                selected_idxs = lb_idxs, weak_transform=weak,normal_toTensor=normal_toTensor)
            ulb_dataset = ProstateSegmentation(base_dir=train_data_path, phase='train', splitid=lb_domain, domain=domain, 
                                                selected_idxs=unlabeled_idxs, weak_transform=weak, strong_tranform=strong,normal_toTensor=normal_toTensor)
        for i in range(1,domain_num+1):
            cur_dataset = ProstateSegmentation(base_dir=train_data_path, phase='test', splitid=-1, domain=[i], normal_toTensor=normal_toTensor)
            test_dataset.append(cur_dataset)
    elif args.dataset == 'MNMS':
        if not args.eval:
            lb_dataset = MNMSSegmentation(base_dir=train_data_path, phase='train', splitid=lb_domain, domain=[lb_domain], 
                                                selected_idxs = lb_idxs, weak_transform=weak,normal_toTensor=normal_toTensor)
            ulb_dataset = MNMSSegmentation(base_dir=train_data_path, phase='train', splitid=lb_domain, domain=domain, 
                                                selected_idxs=unlabeled_idxs, weak_transform=weak, strong_tranform=strong,normal_toTensor=normal_toTensor)
        for i in range(1,domain_num+1):
            cur_dataset = MNMSSegmentation(base_dir=train_data_path, phase='test', splitid=-1, domain=[i], normal_toTensor=normal_toTensor)
            test_dataset.append(cur_dataset)
    if not args.eval:
        lb_dataloader = cycle(DataLoader(lb_dataset, batch_size = args.label_bs, shuffle=True, num_workers=2, pin_memory=True, drop_last=False))
        ulb_dataloader = cycle(DataLoader(ulb_dataset, batch_size = args.unlabel_bs, shuffle=True, num_workers=2, pin_memory=True, drop_last=False))
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


    # if restoring previous models:
    if args.load:
        try:
            # check if there is previous progress to be restored:
            logging.info(f"Snapshot path: {snapshot_path}")
            iter_num = []
            for filename in os.listdir(snapshot_path):
                if "model_iter" in filename:
                    basename, extension = os.path.splitext(filename)
                    iter_num.append(int(basename.split("_")[2]))
            iter_num = max(iter_num)
            for filename in os.listdir(snapshot_path):
                if "model_iter" in filename and str(iter_num) in filename:
                    model_checkpoint = filename
        except Exception as e:
            logging.warning(f"Error finding previous checkpoints: {e}")

        try:
            logging.info(f"Restoring model checkpoint: {model_checkpoint}")
            model, optimizer, start_epoch, performance = util.load_checkpoint(
                snapshot_path + "/" + model_checkpoint, model, optimizer
            )
            logging.info(f"Models restored from iteration {iter_num}")
        except Exception as e:
            logging.warning(f"Unable to restore model checkpoint: {e}, using new model")

    # set to train

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

    logging.info("{} iterations per epoch".format(args.num_eval_iter))

    max_epoch = max_iterations // args.num_eval_iter
    if args.dataset == 'fundus':
        best_cup_dice, best_disc_dice = 0.0, 0.0
        best_cup_dice_iter, best_disc_dice_iter = -1, -1
        stu_best_cup_dice, stu_best_disc_dice = 0.0, 0.0
        stu_best_cup_dice_iter, stu_best_disc_dice_iter = -1, -1
    elif args.dataset == 'prostate':
        best_dice = 0.0
        best_dice_iter = -1
        stu_best_dice = 0.0
        stu_best_dice_iter = -1
    elif args.dataset == 'MNMS':
        best_lv_dice, best_myo_dice, best_rv_dice = 0.0, 0.0, 0.0
        best_lv_dice_iter, best_myo_dice_iter, best_rv_dice_iter = -1, -1, -1
        stu_best_lv_dice, stu_best_myo_dice , stu_best_rv_dice= 0.0, 0.0, 0.0
        stu_best_lv_dice_iter, stu_best_myo_dice_iter, stu_best_rv_dice_iter = -1, -1, -1

    iter_num = int(iter_num)

    # iterator = tqdm(range(start_epoch, max_epoch), ncols=70)
    threshold = args.threshold

    if args.eval:
        # for i in range(1,5):
        #     model.load_state_dict(torch.load('../model/lb{}_r0.2_fixmatch_th0.9/unet_disc_dice_best_model.pth'.format(i)))
        #     test(args, model,test_dataloader,i,writer)
        model.load_state_dict(torch.load('../model/prostate/pu_0.9probfda_lb{}_r0.2_th0.9_v2/unet_dice_best_model.pth'.format(args.lb_domain)))
        test(args, model,test_dataloader,args.lb_domain,writer)
        exit()

    scaler = GradScaler()
    amp_cm = autocast if args.amp else contextlib.nullcontext


    for epoch_num in range(start_epoch, max_epoch):
        model.train()
        ema_model.train()
        p_bar = tqdm(range(args.num_eval_iter))
        p_bar.set_description(f'No. {epoch_num+1}')
        for i_batch in range(1, args.num_eval_iter+1):
            lb_sample = next(lb_dataloader)
            ulb_sample = next(ulb_dataloader)
            lb_x_w, lb_y = lb_sample['image'], lb_sample['label']
            ulb_x_w, ulb_x_s, ulb_y = ulb_sample['image'], ulb_sample['strong_aug'], ulb_sample['label']
            lb_dc, ulb_dc = lb_sample['dc'].cuda(), ulb_sample['dc'].cuda()
            ulb_dc_type = ulb_dc.clone()
            inlier = ulb_dc == lb_dc[0]
            outlier = ulb_dc != lb_dc[0]

            lb_x_w, lb_y, ulb_x_w, ulb_x_s, ulb_y = lb_x_w.cuda(), lb_y.cuda(), ulb_x_w.cuda(), ulb_x_s.cuda(), ulb_y.cuda()
            if args.dataset == 'fundus':
                lb_cup_label = lb_y.eq(0).float()  # == 0
                lb_disc_label = lb_y.le(128).float()  # <= 128
                lb_mask = torch.cat((lb_cup_label, lb_disc_label),dim=1)
                ulb_cup_label = ulb_y.eq(0).float()
                ulb_disc_label = ulb_y.le(128).float()
                ulb_mask = torch.cat((ulb_cup_label, ulb_disc_label),dim=1)
                lb_mask_shape = [len(lb_x_w), num_classes, patch_size, patch_size]
                ulb_mask_shape = [len(ulb_x_w), num_classes, patch_size, patch_size]
            elif args.dataset == 'prostate':
                lb_mask = lb_y.eq(0).float()
                ulb_mask = ulb_y.eq(0).float()
                lb_mask_shape = [len(lb_x_w), num_classes, patch_size, patch_size]
                ulb_mask_shape = [len(ulb_x_w), num_classes, patch_size, patch_size]
            elif args.dataset == 'MNMS':
                lb_lv_label = lb_y[:,0,...].eq(255).float()
                lb_myo_label = lb_y[:,1,...].eq(255).float()
                lb_rv_label = lb_y[:,2,...].eq(255).float()
                lb_back_label = 1-(lb_lv_label+lb_myo_label+lb_rv_label)
                lb_mask = torch.stack((lb_lv_label, lb_myo_label, lb_rv_label, lb_back_label),dim=1)
                ulb_lv_label = ulb_y[:,0,...].eq(255).float()
                ulb_myo_label = ulb_y[:,1,...].eq(255).float()
                ulb_rv_label = ulb_y[:,2,...].eq(255).float()
                ulb_back_label = 1-(ulb_lv_label+ulb_myo_label+ulb_rv_label)
                ulb_mask = torch.stack((ulb_lv_label, ulb_myo_label, ulb_rv_label, ulb_back_label),dim=1)
                lb_mask_shape = [len(lb_x_w), 1, patch_size, patch_size]
                ulb_mask_shape = [len(ulb_x_w), 1, patch_size, patch_size]

            with amp_cm():

                cutmix_box = torch.stack([obtain_cutmix_box(img_size=patch_size, p=args.cutmix_prob) for i in range(len(ulb_x_s))], dim=0)
                choice = np.random.randint(0, len(lb_x_w), (1, len(ulb_x_s)))
                mix_img = lb_x_w[choice]
                ulb_x_s[cutmix_box.unsqueeze(1).expand(ulb_x_s.shape) == 1] = mix_img[cutmix_box.unsqueeze(1).expand(ulb_x_s.shape) == 1]

                # outputs for model
                logits_lb_x_w = model(lb_x_w)
                logits_ulb_x_w = ema_model(ulb_x_w)
                logits_ulb_x_s = model(ulb_x_s)
                if args.dataset == 'fundus' or args.dataset == 'prostate':
                    prob_lb_x_w = logits_lb_x_w.sigmoid()
                    prob_ulb_x_w = logits_ulb_x_w.sigmoid()
                    prob_ulb_x_s = logits_ulb_x_s.sigmoid()
                    pseudo_label = prob_ulb_x_w.ge(0.5).float().detach()
                elif args.dataset == 'MNMS':
                    prob_lb_x_w = F.softmax(logits_lb_x_w, dim=1)
                    max_prob, max_idx = torch.max(prob_lb_x_w,dim=1, keepdim=True)
                    lb_pred = (prob_lb_x_w == max_prob).float().detach()
                    prob_ulb_x_w = F.softmax(logits_ulb_x_w, dim=1)
                    prob_ulb_x_s = F.softmax(logits_ulb_x_s, dim=1)
                    max_prob, max_idx = torch.max(prob_ulb_x_w,dim=1, keepdim=True)
                    pseudo_label = (prob_ulb_x_w == max_prob).float().detach()


                if args.dataset == 'fundus':
                    ulb_cup_dice, ulb_disc_dice = metrics.dice_coeff_2label(np.asarray(pseudo_label.clone().cpu()), ulb_mask.clone().cpu())
                elif args.dataset == 'prostate':
                    ulb_dice = metrics.dice_coeff(np.asarray(pseudo_label.clone().cpu()), ulb_mask.clone().cpu())
                elif args.dataset == 'MNMS':
                    ulb_lv_dice, ulb_myo_dice, ulb_rv_dice = metrics.dice_coeff_3label(np.asarray(pseudo_label.clone().cpu()), ulb_mask.clone().cpu())

                if args.dataset == 'fundus' or args.dataset == 'prostate':
                    sup_loss = bce_loss(logits_lb_x_w, lb_mask).mean()
                elif args.dataset == 'MNMS':
                    sup_loss = F.cross_entropy(logits_lb_x_w, torch.argmax(lb_mask, dim=1), reduction='none').mean()

                consistency_weight = get_current_consistency_weight(
                    iter_num // (args.max_iterations/args.consistency_rampup))

                if args.dataset == 'fundus' or args.dataset == 'prostate':
                    mask = prob_ulb_x_w.ge(threshold).float() + prob_ulb_x_w.le(1-threshold).float()
                elif args.dataset == 'MNMS':
                    max_prob, max_idx = torch.max(prob_ulb_x_w,dim=1, keepdim=True)
                    mask = max_prob.ge(threshold).float()
                
                mask[cutmix_box.unsqueeze(1).expand(ulb_mask_shape) == 1] = 1
                pseudo_label[cutmix_box.unsqueeze(1).expand(pseudo_label.shape) == 1] = lb_mask[choice][cutmix_box.unsqueeze(1).expand(pseudo_label.shape) == 1]
                if args.dataset == 'fundus' or args.dataset == 'prostate':
                    unsup_loss = (bce_loss(logits_ulb_x_s, pseudo_label) * mask).mean()
                elif args.dataset == 'MNMS':
                    unsup_loss = (F.cross_entropy(logits_ulb_x_s, torch.argmax(pseudo_label, dim=1), reduction='none').unsqueeze(1) * mask).mean()
                
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
            if args.dataset == 'fundus':
                writer.add_scalar('train/ulb_cup_dice', ulb_cup_dice, iter_num)
                writer.add_scalar('train/ulb_disc_dice', ulb_disc_dice, iter_num)
            elif args.dataset == 'prostate':
                writer.add_scalar('train/ulb_dice', ulb_dice, iter_num)
            elif args.dataset == 'MNMS':
                writer.add_scalar('train/ulb_lv_dice', ulb_lv_dice, iter_num)
                writer.add_scalar('train/ulb_myo_dice', ulb_myo_dice, iter_num)
                writer.add_scalar('train/ulb_rv_dice', ulb_rv_dice, iter_num)
            writer.add_scalar('train/mask', mask.mean(), iter_num)
            writer.add_scalar('train/lr', lr_, iter_num)
            writer.add_scalar('train/loss', loss.item(), iter_num)
            writer.add_scalar('train/sup_loss', sup_loss.item(), iter_num)
            writer.add_scalar('train/unsup_loss', unsup_loss.item(), iter_num)
            writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
            if p_bar is not None:
                p_bar.update()

            if args.dataset == 'fundus':
                p_bar.set_description('iteration %d : loss:%f, sup_loss:%f, unsup_loss:%f, cons_w:%f, mask_ratio:%f, ulb_cd:%f, ulb_dd:%f' 
                                        % (iter_num, loss.item(), sup_loss.item(), unsup_loss.item(), consistency_weight, mask.mean(), ulb_cup_dice, ulb_disc_dice))
            elif args.dataset == 'prostate':
                p_bar.set_description('iteration %d : loss:%f, sup_loss:%f, unsup_loss:%f, cons_w:%f, mask_ratio:%f, ulb_dice:%f' 
                                    % (iter_num, loss.item(), sup_loss.item(), unsup_loss.item(), consistency_weight, mask.mean(), ulb_dice))
            elif args.dataset == 'MNMS':
                p_bar.set_description('iteration %d: loss:%.3f, sup_loss:%.3f, unsup_loss:%.3f, cons_w:%.3f, mask_ratio:%.3f, ul:%.3f, um:%.3f, ur:%.3f' 
                                        % (iter_num, loss.item(), sup_loss.item(), unsup_loss.item(), consistency_weight, mask.mean(), ulb_lv_dice, ulb_myo_dice, ulb_rv_dice))
            if iter_num % 200 == 0:
            # if cm_flag and pred[0] == 0:
                # logging.info('draw confidence-dice table...')
                # confidence_dice.append([mask.mean(), ulb_dice])
                # idx = int((iter_num-1)/(max_iterations/3))
                # ax.scatter(mask.mean().item(), ulb_dice, c = color_list[idx], s = 16, alpha=0.3)
                # plt.savefig(img_save_path)
                # logging.info('record train img...')
                if args.dataset == 'fundus':
                    lb_image = make_grid([make_grid(lb_x_w[0, ...].clone().cpu().data,1,normalize=True), lb_mask[0,0,...].clone().unsqueeze(0).repeat(3,1,1).cpu().data, 
                                            logits_lb_x_w.sigmoid()[0, 0, ...].clone().unsqueeze(0).repeat(3,1,1).ge(0.5).float().cpu().data, 
                                            lb_mask[0, 1, ...].clone().unsqueeze(0).repeat(3,1,1).cpu().data, 
                                            logits_lb_x_w.sigmoid()[0, 1, ...].clone().unsqueeze(0).repeat(3,1,1).ge(0.5).float().cpu().data],5,padding=2, pad_value=1)
                    ulb_image = make_grid([make_grid(ulb_x_w[0, ...].clone().cpu().data,1,normalize=True), ulb_mask[0,0,...].clone().unsqueeze(0).repeat(3,1,1).cpu().data, 
                                            ulb_mask[0, 1, ...].clone().unsqueeze(0).repeat(3,1,1).cpu().data, 
                                            make_grid(ulb_x_s[0, ...].clone().cpu().data,1,normalize=True),
                                            pseudo_label[0, 0, ...].clone().unsqueeze(0).repeat(3,1,1).cpu().data, 
                                            pseudo_label[0, 1, ...].clone().unsqueeze(0).repeat(3,1,1).cpu().data],3,padding=2, pad_value=1)
                elif args.dataset == 'prostate':
                    lb_image = make_grid([make_grid(lb_x_w[0, ...].clone().cpu().data,1,normalize=True), lb_mask[0,...].clone().repeat(3,1,1).cpu().data, 
                                            logits_lb_x_w.sigmoid()[0, ...].clone().repeat(3,1,1).ge(0.5).float().cpu().data],3,padding=2, pad_value=1)
                    ulb_image = make_grid([make_grid(ulb_x_w[0, ...].clone().cpu().data,1,normalize=True), ulb_mask[0,...].clone().repeat(3,1,1).cpu().data, 
                                            make_grid(ulb_x_s[0, ...].clone().cpu().data,1,normalize=True), pseudo_label[0, ...].clone().repeat(3,1,1).cpu().data],4,padding=2, pad_value=1)
                elif args.dataset == 'MNMS':
                    lb_image = make_grid([make_grid(lb_x_w[0, :3, ...].clone().cpu().data,1,normalize=True), 
                                            lb_mask[0,:3, ...].clone().cpu().data, 
                                            lb_pred[0, :3,...].clone().cpu().data],3,padding=2, pad_value=1)
                    ulb_image = make_grid([make_grid(ulb_x_w[0, :3, ...].clone().cpu().data,1,normalize=True), 
                                            ulb_mask[0, :3,...].clone().cpu().data, 
                                            make_grid(ulb_x_s[0, :3, ...].clone().cpu().data,1,normalize=True),
                                            pseudo_label[0, :3,...].clone().cpu().data],2,padding=2, pad_value=1)
                writer.add_image("train/lb_sample", lb_image, iter_num)
                writer.add_image("train/ulb_sample", ulb_image, iter_num)
                logging.info('iteration %d : loss : %f, sup_loss : %f, unsup_loss : %f, cons_w : %f, mask_ratio : %f' 
                                    % (iter_num, loss.item(), sup_loss.item(), unsup_loss.item(), consistency_weight, mask.mean()))
                if args.dataset == 'fundus':
                    logging.info('ulb_cd:%f, ulb_dd:%f' 
                                            % (ulb_cup_dice, ulb_disc_dice))
                elif args.dataset == 'prostate':
                    logging.info('ulb_dice:%f' 
                                        % (ulb_dice))
                elif args.dataset == 'MNMS':
                    logging.info('ulb_lv:%f, ulb_myo:%f, ulb_rv:%f' 
                                            % (ulb_lv_dice, ulb_myo_dice, ulb_rv_dice))
                    
        
        if p_bar is not None:
            p_bar.close()

        if args.dataset == 'fundus':
            logging.info('test ema model')
            val_cup_dice, val_disc_dice = test(args, ema_model, test_dataloader, epoch_num+1, writer)
            if iter_num == max_iterations:
                cur_save_path = os.path.join(
                    snapshot_path,
                    "iter_{}_cup_dice_{}_disc_dice_{}.pth".format(iter_num, round(val_cup_dice, 4), round(val_disc_dice, 4)),
                )
                logging.info('save cur model to {}'.format(cur_save_path))
                torch.save(ema_model.state_dict(), cur_save_path)
                torch.save(ema_model.state_dict(), cur_save_path)
            if val_cup_dice > best_cup_dice:
                best_cup_dice = val_cup_dice
                best_cup_dice_iter = iter_num
                save_best = os.path.join(snapshot_path, "{}_cup_dice_best_model.pth".format(args.model))
                # util.save_checkpoint(epoch_num, model, optimizer, loss, save_mode_path)
                # util.save_checkpoint(epoch_num, model, optimizer, loss, save_best)
                logging.info('save cur best cup model to {}'.format(save_best))
                torch.save(ema_model.state_dict(), save_best)
            if val_disc_dice > best_disc_dice:
                best_disc_dice = val_disc_dice
                best_disc_dice_iter = iter_num
                save_best = os.path.join(snapshot_path, "{}_disc_dice_best_model.pth".format(args.model))
                # util.save_checkpoint(epoch_num, model, optimizer, loss, save_mode_path)
                # util.save_checkpoint(epoch_num, model, optimizer, loss, save_best)
                logging.info('save cur best disc model to {}'.format(save_best))
                torch.save(ema_model.state_dict(), save_best)
            logging.info('val_cup_best_dice: %f at %d iter, val_disc_best_dice: %f at %d iter' %
                    (best_cup_dice,best_cup_dice_iter,best_disc_dice,best_disc_dice_iter))
            
            if args.test_stu:
                logging.info('test stu model')
                stu_val_cup_dice, stu_val_disc_dice = test(args, model, test_dataloader, epoch_num+1, writer, ema=False)
                if stu_val_cup_dice > stu_best_cup_dice:
                    stu_best_cup_dice = stu_val_cup_dice
                    stu_best_cup_dice_iter = iter_num
                if stu_val_disc_dice > stu_best_disc_dice:
                    stu_best_disc_dice = stu_val_disc_dice
                    stu_best_disc_dice_iter = iter_num
                logging.info('stu_val_cup_best_dice: %f at %d iter, stu_val_disc_best_dice: %f at %d iter' %
                        (stu_best_cup_dice,stu_best_cup_dice_iter,stu_best_disc_dice,stu_best_disc_dice_iter))
        elif args.dataset == 'prostate':
            val_dice = test(args, ema_model, test_dataloader, epoch_num+1, writer)
            if iter_num == max_iterations:
                cur_save_path = os.path.join(
                    snapshot_path,
                    "iter_{}_dice_{}.pth".format(iter_num, round(val_dice, 4)),
                )
                logging.info('save cur model to {}'.format(cur_save_path))
                torch.save(ema_model.state_dict(), cur_save_path)
                torch.save(ema_model.state_dict(), cur_save_path)
            if val_dice > best_dice:
                best_dice = val_dice
                best_dice_iter = iter_num
                save_best = os.path.join(snapshot_path, "{}_dice_best_model.pth".format(args.model))
                # util.save_checkpoint(epoch_num, model, optimizer, loss, save_mode_path)
                # util.save_checkpoint(epoch_num, model, optimizer, loss, save_best)
                logging.info('save cur best model to {}'.format(save_best))
                torch.save(ema_model.state_dict(), save_best)
            logging.info('val_best_dice: %f at %d iter' %
                    (best_dice,best_dice_iter))
            
            if args.test_stu:
                logging.info('test stu model')
                stu_val_dice = test(args, model, test_dataloader, epoch_num+1, writer, ema=False)
                if stu_val_dice > stu_best_dice:
                    stu_best_dice = stu_val_dice
                    stu_best_dice_iter = iter_num
                logging.info('stu_val_best_dice: %f at %d iter' %
                        (stu_best_dice,stu_best_dice_iter))
        elif args.dataset == 'MNMS':
            logging.info('test ema model')
            val_lv_dice, val_myo_dice, val_rv_dice = test(args, ema_model, test_dataloader, epoch_num+1, writer)
            if iter_num == max_iterations:
                cur_save_path = os.path.join(
                    snapshot_path,
                    "iter_{}_lv_dice_{}_myo_dice_{}_rv_dice_{}.pth".format(iter_num, round(val_lv_dice, 4), round(val_myo_dice, 4), round(val_rv_dice, 4)),
                )
                logging.info('save cur model to {}'.format(cur_save_path))
                torch.save(ema_model.state_dict(), cur_save_path)
                torch.save(ema_model.state_dict(), cur_save_path)
            if val_lv_dice > best_lv_dice:
                best_lv_dice = val_lv_dice
                best_lv_dice_iter = iter_num
                save_best = os.path.join(snapshot_path, "{}_lv_dice_best_model.pth".format(args.model))
                logging.info('save cur best lv model to {}'.format(save_best))
                torch.save(ema_model.state_dict(), save_best)
            if val_myo_dice > best_myo_dice:
                best_myo_dice = val_myo_dice
                best_myo_dice_iter = iter_num
                save_best = os.path.join(snapshot_path, "{}_myo_dice_best_model.pth".format(args.model))
                logging.info('save cur best myo model to {}'.format(save_best))
                torch.save(ema_model.state_dict(), save_best)
            if val_rv_dice > best_rv_dice:
                best_rv_dice = val_rv_dice
                best_rv_dice_iter = iter_num
                save_best = os.path.join(snapshot_path, "{}_rv_dice_best_model.pth".format(args.model))
                logging.info('save cur best rv model to {}'.format(save_best))
                torch.save(ema_model.state_dict(), save_best)
            logging.info('val_lv_best_dice: %f at %d iter, val_myo_best_dice: %f at %d iter, val_rv_best_dice: %f at %d iter' %
                    (best_lv_dice,best_lv_dice_iter,best_myo_dice,best_myo_dice_iter,best_rv_dice,best_rv_dice_iter))
            
            if args.test_stu:
                logging.info('test stu model')
                stu_val_lv_dice, stu_val_myo_dice, stu_val_rv_dice = test(args, model, test_dataloader, epoch_num+1, writer, ema=False)
                if stu_val_lv_dice > stu_best_lv_dice:
                    stu_best_lv_dice = stu_val_lv_dice
                    stu_best_lv_dice_iter = iter_num
                if stu_val_myo_dice > stu_best_myo_dice:
                    stu_best_myo_dice = stu_val_myo_dice
                    stu_best_myo_dice_iter = iter_num
                if stu_val_rv_dice > stu_best_rv_dice:
                    stu_best_rv_dice = stu_val_rv_dice
                    stu_best_rv_dice_iter = iter_num
                logging.info('stu_val_lv_best_dice: %f at %d iter, stu_val_myo_best_dice: %f at %d iter, stu_val_rv_best_dice: %f at %d iter' %
                        (stu_best_lv_dice,stu_best_lv_dice_iter,stu_best_myo_dice,stu_best_myo_dice_iter,stu_best_rv_dice,stu_best_rv_dice_iter))

        
    writer.close()


if __name__ == "__main__":
    snapshot_path = "../model/" + args.dataset + "/" + args.save_name + "/"
    if args.dataset == 'fundus':
        train_data_path='../../data/Fundus'
    elif args.dataset == 'prostate':
        train_data_path="../../data/ProstateSlice"
    elif args.dataset == 'MNMS':
        train_data_path="../../data/MNMS/mnms_split_2D_ROI"

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
