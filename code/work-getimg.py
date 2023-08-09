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
from PIL import Image
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='fundus', choices=['fundus', 'prostate'])
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
parser.add_argument("--test_stu", default=True, action='store_true')
args = parser.parse_args()

def create_model(ema=False):
        # Network definition
        if args.model == 'unet':
            model = UNet(n_channels = 3, n_classes = 2)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model.cuda()

model = create_model()

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

    if args.dataset == 'fundus':
        patch_size = 256
        min_v, max_v = 0.5, 1.5
        fillcolor = 255
        num_channels = 3
    elif args.dataset == 'prostate':
        patch_size = 384
        min_v, max_v = 0.1, 2
        fillcolor = 255
    
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

    img_path = os.path.join('../img/fundus/4/u_c.png')
    x = Image.open(img_path).convert('RGB').resize((256, 256), Image.LANCZOS)
    y = Image.open(img_path.replace('image', 'mask'))
    if y.mode is 'RGB':
        y = y.convert('L')
    y = y.resize((256, 256), Image.NEAREST)
    model.load_state_dict(torch.load('../model/'+args.dataset+'/20plmaskcutmix1.0005_lb1/unet_cup_dice_best_model.pth'))
    train_sample = {'image': np.array(x), 'label':np.array(y)}
    train_sample = normal_toTensor(train_sample)
    train_x = train_sample['image'].unsqueeze(0).cuda()
    pred = model(train_x)[0].sigmoid().ge(0.5).float().cpu()
    pred_img = torch.zeros(plc.shape)
    pred_img[pred[1] == 1] = 128
    pred_img[pred[0] == 1] = 0
    pred_img = pred_img.numpy()
    cv2.imwrite(snapshot_path+'pred.png', np.array(pred_img).astype(np.uint8))
    exit()

    domain_name = {1:'Domain1', 2:'Domain2', 3:'Domain3', 4:'Domain4'}
    domain = 4
    img_name = 'V0029.png'
    base_dir='/data/qinghe/data/Fundus'
    img_path = os.path.join(base_dir, domain_name[domain], 'train/ROIs/image/',img_name)
    
    
    img = Image.open(img_path).convert('RGB').resize((256, 256), Image.LANCZOS)
    target = Image.open(img_path.replace('image', 'mask'))
    if target.mode is 'RGB':
        target = target.convert('L')
    target = target.resize((256, 256), Image.NEAREST)

    sample = {'image': img, 'label': target, 'img_name': img_name, 'dc': domain}
    sample = weak(sample)
    weak_img = sample['image']
    strong_img = strong(weak_img)

    # plt.imshow((np.array(img)).astype(np.uint8))
    # plt.savefig(snapshot_path+'ori.png')
    # plt.cla()
    # plt.imshow((np.array(sample['image'])).astype(np.uint8))
    # plt.savefig(snapshot_path+'weakimg.png')
    # plt.cla()
    # plt.imshow((np.array(strong_img)).astype(np.uint8))
    # plt.savefig(snapshot_path+'strongimg.png')
    # plt.cla()
    # plt.imshow(np.array(sample['label']).astype(np.uint8),cmap='Greys_r')
    # plt.savefig(snapshot_path+'weakmask.png')
    # plt.cla()
    cv2.imwrite(snapshot_path+'ori.png', cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
    cv2.imwrite(snapshot_path+'weakimg.png', cv2.cvtColor(np.array(sample['image']), cv2.COLOR_RGB2BGR))
    cv2.imwrite(snapshot_path+'strongimg.png', cv2.cvtColor(np.array(strong_img), cv2.COLOR_RGB2BGR))
    cv2.imwrite(snapshot_path+'weakmask.png', np.array(sample['label']))

    domain = 1
    img_name = 'ndrishtiGS_035.png'
    img_path = os.path.join(base_dir, domain_name[domain], 'train/ROIs/image/',img_name)
    x = Image.open(img_path).convert('RGB').resize((256, 256), Image.LANCZOS)
    y = Image.open(img_path.replace('image', 'mask'))
    if y.mode is 'RGB':
        y = y.convert('L')
    y = y.resize((256, 256), Image.NEAREST)
    x_sample = {'image': x, 'label': y, 'img_name': img_name, 'dc': domain}
    x_sample = weak(x_sample)
    x_weak_img = x_sample['image']
    # plt.imshow((np.array(x)).astype(np.uint8))
    # plt.savefig(snapshot_path+'x_ori.png')
    # plt.cla()
    # plt.imshow((np.array(x_sample['image'])).astype(np.uint8))
    # plt.savefig(snapshot_path+'x_weakimg.png')
    # plt.cla()
    # plt.imshow(np.array(x_sample['label']).astype(np.uint8),cmap='Greys_r')
    # plt.savefig(snapshot_path+'x_weakmask.png')
    # plt.cla()

    cv2.imwrite(snapshot_path+'x_ori.png', cv2.cvtColor(np.array(x), cv2.COLOR_RGB2BGR))
    cv2.imwrite(snapshot_path+'x_weakimg.png', cv2.cvtColor(np.array(x_sample['image']), cv2.COLOR_RGB2BGR))
    cv2.imwrite(snapshot_path+'x_weakmask.png', np.array(x_sample['label']))

    box = obtain_cutmix_box(patch_size, 1)
    # plt.imshow(np.array(box).astype(np.uint8),cmap='Greys_r')
    # plt.savefig(snapshot_path+'box.png')
    # plt.cla()
    
    cv2.imwrite(snapshot_path+'box.png', np.array(box*255).astype(np.uint8))

    uc = np.array(strong_img).copy()
    uc[box==1] = np.array(x_weak_img)[box==1]
    plc = np.array(sample['label']).copy()
    plc[box==1] = np.array(x_sample['label'])[box==1]
    plt.imshow((np.array(uc)).astype(np.uint8))
    plt.savefig(snapshot_path+'u_c.png')
    plt.cla()
    plt.imshow(np.array(plc).astype(np.uint8),cmap='Greys_r')
    plt.savefig(snapshot_path+'plc.png')
    plt.cla()

    cv2.imwrite(snapshot_path+'u_c.png', cv2.cvtColor(np.array(uc).astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(snapshot_path+'plc.png', np.array(plc).astype(np.uint8))

    model.load_state_dict(torch.load('../model/'+args.dataset+'/20plmaskcutmix1.0005_lb1/unet_cup_dice_best_model.pth'))
    train_sample = {'image': np.array(uc), 'label':np.array(plc)}
    train_sample = normal_toTensor(train_sample)
    train_x = train_sample['image'].unsqueeze(0)
    pred = model(train_x)[0].sigmoid().ge(0.5).float()
    pred_img = torch.zeros(plc.shape)
    pred_img[pred[1] == 1] = 128
    pred_img[pred[0] == 1] = 0
    pred_img = pred_img.numpy()
    cv2.imwrite(snapshot_path+'pred.png', np.array(pred_img).astype(np.uint8))
            


if __name__ == "__main__":
    
    save_parent_path = "../img/" + args.dataset
    if not os.path.exists(save_parent_path):
        os.makedirs(save_parent_path)
    listd= os.listdir("../img/" + args.dataset)
    num = len(listd)
    snapshot_path = "../img/" + args.dataset + "/" + str(num) + "/"
    if args.dataset == 'fundus':
        train_data_path='../../data/Fundus'
    elif args.dataset == 'prostate':
        train_data_path="../../data/ProstateSlice"
        
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    train(args, snapshot_path)
