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
            model = UNet(n_channels = 3, n_classes = 1)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model.cuda()

model = create_model()

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

    domain_name = {1:'Domain1', 2:'Domain2', 3:'Domain3', 4:'Domain4'}
    domain = 1
    img_name = 'ndrishtiGS_046.png'
    base_dir='/data/qinghe/data/Fundus'
    img_path = os.path.join(base_dir, domain_name[domain], 'train/ROIs/image/',img_name)
    
    
    img = Image.open(img_path).convert('RGB').resize((256, 256), Image.LANCZOS)
    target = Image.open(img_path.replace('image', 'mask'))
    if target.mode is 'RGB':
        target = target.convert('L')
    target = target.resize((256, 256), Image.NEAREST)

    sample = {'image': img, 'label': target, 'img_name': img_name}
    sample = weak(sample)
    weak_img = sample['image']
    strong_img = strong(weak_img)

    plt.imshow((np.array(img)).astype(np.uint8))
    plt.savefig(snapshot_path+'ori.png')
    plt.cla()
    plt.imshow((np.array(sample['image'])).astype(np.uint8))
    plt.savefig(snapshot_path+'weakimg.png')
    plt.cla()
    plt.imshow((np.array(strong_img)).astype(np.uint8))
    plt.savefig(snapshot_path+'strongimg.png')
    plt.cla()
    plt.imshow(np.array(sample['label']).astype(np.uint8),cmap='Greys_r')
    plt.savefig(snapshot_path+'weakmask.png')
    plt.cla()



            


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

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    train(args, snapshot_path)
