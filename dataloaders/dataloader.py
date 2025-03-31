from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from glob import glob
import random
import copy
import matplotlib.pyplot as plt
from dataloaders.transform import crop, hflip, normalize, resize, blur, cutout
from torchvision import transforms

class FundusSegmentation(Dataset):
    """
    Fundus segmentation dataset
    including 5 domain dataset
    one for test others for training
    """

    def __init__(self,
                 base_dir='/data/qinghe/data/Fundus',
                 phase='train',
                 splitid=2,
                 domain=[1,2,3,4],
                 weak_transform=None,
                 strong_tranform=None,
                 normal_toTensor = None,
                 selected_idxs = None
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        # super().__init__()
        self._base_dir = base_dir
        self.image_list = []
        self.phase = phase
        self.domain_name = {1:'DGS', 2:'RIM', 3:'REF', 4:'REF_val'}
        self.image_pool = []
        self.label_pool = []
        self.img_name_pool = []
        self.img_domain_code_pool = []

        self.flags_DGS = ['gd', 'nd']
        self.flags_REF = ['g', 'n']
        self.flags_RIM = ['G', 'N', 'S']
        self.flags_REF_val = ['V']
        self.splitid = splitid
        self.domain = domain
        SEED = 1212
        random.seed(SEED)
        excluded_num = 0
        for i in self.domain:
            self._image_dir = os.path.join(self._base_dir, 'Domain'+str(i), phase, 'ROIs/image/')
            print('==> Loading {} data from: {}'.format(phase, self._image_dir))

            if phase == 'train':
                imagelist = []
                with open(os.path.join(self._base_dir, f'Domain{i}_train.txt')) as f:
                    for line in f:
                        imagelist.append(line.strip())
            else:
                imagelist = glob(self._image_dir + '*.png')
                imagelist.sort()

            if self.splitid == i and selected_idxs is not None:
                total = list(range(len(imagelist)))
                excluded_idxs = [x for x in total if x not in selected_idxs]
                excluded_num += len(excluded_idxs)
            else:
                excluded_idxs = []
            
            for exclude_id in reversed(sorted(excluded_idxs)):
                imagelist.pop(exclude_id)
                
            for image_path in imagelist:
                self.image_pool.append(image_path)
                gt_path = image_path.replace('image', 'mask')
                self.label_pool.append(gt_path)
                self.img_domain_code_pool.append(i)
                _img_name = image_path.split('/')[-1]
                self.img_name_pool.append(_img_name)

        self.weak_transform = weak_transform
        self.strong_transform = strong_tranform
        self.normal_toTensor = normal_toTensor
        
        
        print('-----Total number of images in {}: {:d}, Excluded: {:d}'.format(phase, len(self.image_pool), excluded_num))

    def __len__(self):
        return len(self.image_pool)
        
    def __getitem__(self, index):
        if self.phase != 'test':
            _img = Image.open(self.image_pool[index]).convert('RGB').resize((256, 256), Image.LANCZOS)
            _target = Image.open(self.label_pool[index])
            if _target.mode is 'RGB':
                _target = _target.convert('L')
            _target = _target.resize((256, 256), Image.NEAREST)
            anco_sample = {'image': _img, 'label': _target, 'img_name':self.img_name_pool[index], 'dc': self.img_domain_code_pool[index]}
            if self.weak_transform is not None:
                anco_sample = self.weak_transform(anco_sample)
            if self.strong_transform is not None:
                anco_sample['strong_aug'] = self.strong_transform(anco_sample['image'])
            anco_sample = self.normal_toTensor(anco_sample)
        else:
            _img = Image.open(self.image_pool[index]).convert('RGB').resize((256, 256), Image.LANCZOS)
            _target = Image.open(self.label_pool[index])
            if _target.mode is 'RGB':
                _target = _target.convert('L')
            _target = _target.resize((256, 256), Image.NEAREST)
            anco_sample = {'image': _img, 'label': _target, 'img_name':self.img_name_pool[index], 'dc': self.img_domain_code_pool[index]}
            anco_sample = self.normal_toTensor(anco_sample)
        return anco_sample

    def _read_img_into_memory(self):
        img_num = len(self.image_list)
        for index in range(img_num):
            basename = os.path.basename(self.image_list[index]['image'])
            Flag = "NULL"
            if basename[0:2] in self.flags_DGS:
                Flag = 'DGS'
            elif basename[0] in self.flags_REF:
                Flag = 'REF'
            elif basename[0] in self.flags_RIM:
                Flag = 'RIM'
            elif basename[0] in self.flags_REF_val:
                Flag = 'REF_val'
            else:
                print("[ERROR:] Unknown dataset!")
                return 0
            self.image_pool.append(
                Image.open(self.image_list[index]['image']).convert('RGB').resize((256, 256), Image.LANCZOS))
            _target = Image.open(self.image_list[index]['label'])

            if _target.mode is 'RGB':
                _target = _target.convert('L')
            _target = _target.resize((256, 256), Image.NEAREST)
            self.label_pool.append(_target)
            _img_name = self.image_list[index]['image'].split('/')[-1]
            self.img_name_pool.append(_img_name)
            self.img_domain_code_pool.append(self.image_list[index]['domain_code'])



    def __str__(self):
        return 'Fundus(phase=' + self.phase+str(self.splitid) + ')'

class ProstateSegmentation(Dataset):
    """
    Fundus segmentation dataset
    including 5 domain dataset
    one for test others for training
    """

    def __init__(self,
                 base_dir='/data/qinghe/data//ProstateSlice',
                 phase='train',
                 splitid=2,
                 domain=[1,2,3,4,5,6],
                 weak_transform=None,
                 strong_tranform=None,
                 normal_toTensor = None,
                 selected_idxs = None
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        # super().__init__()
        self._base_dir = base_dir
        self.image_list = []
        self.phase = phase
        self.domain_name = {1:'BIDMC', 2:'BMC', 3:'HK', 4:'I2CVB', 5:'RUNMC', 6:'UCL'}
        self.image_pool = []
        self.label_pool = []
        self.img_name_pool = []
        self.img_domain_code_pool = []

        self.splitid = splitid
        self.domain = domain
        SEED = 1212
        random.seed(SEED)
        excluded_num = 0
        for i in self.domain:
            self._image_dir = os.path.join(self._base_dir, self.domain_name[i], phase,'image/')
            print('==> Loading {} data from: {}'.format(phase, self._image_dir))

            imagelist = glob(self._image_dir + '*.png')
            imagelist.sort()
            if self.splitid == i and selected_idxs is not None:
                total = list(range(len(imagelist)))
                excluded_idxs = [x for x in total if x not in selected_idxs]
                excluded_num += len(excluded_idxs)
            else:
                excluded_idxs = []
            
            for exclude_id in reversed(sorted(excluded_idxs)):
                imagelist.pop(exclude_id)
                
            for image_path in imagelist:
                self.image_pool.append(image_path)
                gt_path = image_path.replace('image', 'mask')
                self.label_pool.append(gt_path)
                self.img_domain_code_pool.append(i)
                _img_name = image_path.split('/')[-1]
                self.img_name_pool.append(self.domain_name[i]+'_'+_img_name)

        self.weak_transform = weak_transform
        self.strong_transform = strong_tranform
        self.normal_toTensor = normal_toTensor
        
        
        print('-----Total number of images in {}: {:d}, Excluded: {:d}'.format(phase, len(self.image_pool), excluded_num))

    def __len__(self):
        return len(self.image_pool)
        
    def __getitem__(self, index):
        if self.phase != 'test':
            _img = Image.open(self.image_pool[index])
            _target = Image.open(self.label_pool[index])
            if _img.mode is 'RGB':
                print('img rgb')
                _img = _img.convert('L')
            if _target.mode is 'RGB':
                print('target rgb')
                _target = _target.convert('L')
            anco_sample = {'image': _img, 'label': _target, 'img_name':self.img_name_pool[index], 'dc': self.img_domain_code_pool[index]}
            if self.weak_transform is not None:
                anco_sample = self.weak_transform(anco_sample)
            if self.strong_transform is not None:
                anco_sample['strong_aug'] = self.strong_transform(anco_sample['image'])
            anco_sample = self.normal_toTensor(anco_sample)
        else:
            _img = Image.open(self.image_pool[index])
            _target = Image.open(self.label_pool[index])
            if _img.mode is 'RGB':
                _img = _img.convert('L')
            if _target.mode is 'RGB':
                _target = _target.convert('L')
            anco_sample = {'image': _img, 'label': _target, 'img_name':self.img_name_pool[index], 'dc': self.img_domain_code_pool[index]}
            anco_sample = self.normal_toTensor(anco_sample)
        return anco_sample




    def __str__(self):
        return 'Prostate(phase=' + self.phase+str(self.splitid) + ')'

class MNMSSegmentation(Dataset):
    """
    MNMS segmentation dataset
    including 5 domain dataset
    one for test others for training
    """

    def __init__(self,
                 base_dir='../../../data/MNMS/mnms_split_2D_ROI',
                 phase='train',
                 splitid=2,
                 domain=[1,2,3,4],
                 weak_transform=None,
                 strong_tranform=None,
                 normal_toTensor = None,
                 selected_idxs = None
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        # super().__init__()
        self._base_dir = base_dir
        self.image_list = []
        self.phase = phase
        self.domain_name = {1:'vendorA', 2:'vendorB', 3:'vendorC', 4:'vendorD'}
        self.image_pool = []
        self.label_pool = []
        self.img_name_pool = []
        self.img_domain_code_pool = []

        self.splitid = splitid
        self.domain = domain
        SEED = 1212
        random.seed(SEED)
        excluded_num = 0
        for i in self.domain:
            self._image_dir = os.path.join(self._base_dir, self.domain_name[i], phase,'image/')
            print('==> Loading {} data from: {}'.format(phase, self._image_dir))

            imagelist = glob(self._image_dir + '*.png')
            imagelist.sort()
            if self.splitid == i and selected_idxs is not None:
                total = list(range(len(imagelist)))
                excluded_idxs = [x for x in total if x not in selected_idxs]
                excluded_num += len(excluded_idxs)
            else:
                excluded_idxs = []
            
            for exclude_id in reversed(sorted(excluded_idxs)):
                imagelist.pop(exclude_id)
                
            for image_path in imagelist:
                self.image_pool.append(image_path)
                gt_path = image_path.replace('image', 'mask')
                self.label_pool.append(gt_path)
                self.img_domain_code_pool.append(i)
                _img_name = image_path.split('/')[-1]
                self.img_name_pool.append(self.domain_name[i]+'_'+_img_name)

        self.weak_transform = weak_transform
        self.strong_transform = strong_tranform
        self.normal_toTensor = normal_toTensor
        
        
        print('-----Total number of images in {}: {:d}, Excluded: {:d}'.format(phase, len(self.image_pool), excluded_num))

    def __len__(self):
        return len(self.image_pool)
        
    def __getitem__(self, index):
        if self.phase != 'test':
            _img = Image.open(self.image_pool[index]).resize((288, 288), Image.BILINEAR)
            _target = Image.open(self.label_pool[index]).resize((288, 288), Image.NEAREST)
            if _img.mode is 'RGB':
                print('img rgb')
                _img = _img.convert('L')
            anco_sample = {'image': _img, 'label': _target, 'img_name':self.img_name_pool[index], 'dc': self.img_domain_code_pool[index]}
            if self.weak_transform is not None:
                anco_sample = self.weak_transform(anco_sample)
            if self.strong_transform is not None:
                anco_sample['strong_aug'] = self.strong_transform(anco_sample['image'])
            anco_sample = self.normal_toTensor(anco_sample)
        else:
            _img = Image.open(self.image_pool[index]).resize((288, 288), Image.BILINEAR)
            _target = Image.open(self.label_pool[index]).resize((288,288), Image.NEAREST)
            if _img.mode is 'RGB':
                _img = _img.convert('L')
            # if _target.mode is 'RGB':
            #     _target = _target.convert('L')
            anco_sample = {'image': _img, 'label': _target, 'img_name':self.img_name_pool[index], 'dc': self.img_domain_code_pool[index]}
            anco_sample = self.normal_toTensor(anco_sample)
        return anco_sample




    def __str__(self):
        return 'MNMS(phase=' + self.phase+str(self.splitid) + ')'

class BUSISegmentation(Dataset):
    def __init__(self, base_dir=None, phase='train', splitid=1, domain=[1,2],
                 weak_transform=None, strong_tranform=None, normal_toTensor=None,
                 selected_idxs = None):
        self._base_dir = base_dir
        self.image_list = []
        self.phase = phase
        self.domain_name = {1:'benign', 2:'malignant'}
        self.sample_list = []
        self.img_name_pool = []
        self.img_domain_code_pool = []
        self.weak_transform = weak_transform
        self.strong_transform = strong_tranform
        self.normal_toTensor = normal_toTensor
        
        self.splitid = splitid
        self.domain = domain
        SEED = 1212
        random.seed(SEED)
        excluded_num = 0
        for i in self.domain:
            self._image_dir = os.path.join(self._base_dir, self.domain_name[i]+'/')
            print('==> Loading {} data from: {}'.format(phase, self._image_dir))

            domain_data_list = []
            imagelist = glob(self._image_dir + '*.png')
            imagelist.sort()
            for image in imagelist:
                if 'mask' not in image:
                    domain_data_list.append([image])
                else:
                    domain_data_list[-1].append(image)
            test_benign_num = int(len(domain_data_list)*0.2)
            train_benign_num = len(domain_data_list) - test_benign_num
            if self.phase == 'test':
                domain_data_list = domain_data_list[-test_benign_num:]
            elif self.phase == 'train':
                domain_data_list = domain_data_list[:train_benign_num]
            else:
                raise Exception('Unknown split...')
            if self.splitid == i and selected_idxs is not None:
                total = list(range(len(domain_data_list)))
                excluded_idxs = [x for x in total if x not in selected_idxs]
                excluded_num += len(excluded_idxs)
            else:
                excluded_idxs = []
            
            for exclude_id in reversed(sorted(excluded_idxs)):
                domain_data_list.pop(exclude_id)
                
            for image_path in domain_data_list:
                self.sample_list.append(image_path)
                self.img_domain_code_pool.append(i)
                _img_name = image_path[0].split('/')[-1]
                self.img_name_pool.append(self.domain_name[i]+'_'+_img_name)
        
        print('-----Total number of images in {}: {:d}, Excluded: {:d}'.format(phase, len(self.sample_list), excluded_num))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        _img = Image.open(self.sample_list[idx][0]).convert('L').resize((256, 256), Image.LANCZOS)
        if len(self.sample_list[idx]) == 2:
            _target = Image.open(self.sample_list[idx][1]).convert('L').resize((256, 256), Image.NEAREST)
        else:
            target_list = []
            for target_path in self.sample_list[idx][1:]:
                target = Image.open(target_path).convert('L')
                target_list.append(np.array(target))
            height, width = target_list[0].shape
            combined_target = np.zeros((height, width), dtype=np.uint8)
            for target in target_list:
                combined_target = np.maximum(combined_target, target)
            if self.phase == 'train':
                _target = Image.fromarray(combined_target).convert('L').resize((256, 256), Image.NEAREST)
            else:
                _target = Image.fromarray(combined_target).convert('L').resize((256, 256), Image.NEAREST)
        
        sample = {'image': _img, 'label': _target, 'img_name':self.img_name_pool[idx], 'dc': self.img_domain_code_pool[idx]}
        if self.phase == "train":
            if self.weak_transform is not None:
                sample = self.weak_transform(sample)
            if self.strong_transform is not None:
                sample['strong_aug'] = self.strong_transform(sample['image'])
            sample = self.normal_toTensor(sample)
        else:
            sample = self.normal_toTensor(sample)
        return sample

def color_map():
    cmap = np.zeros((256, 3), dtype='uint8')
    
    cmap[0] = np.array([128, 64, 128])
    cmap[1] = np.array([244, 35, 232])
    cmap[2] = np.array([70, 70, 70])
    cmap[3] = np.array([102, 102, 156])
    cmap[4] = np.array([190, 153, 153])
    cmap[5] = np.array([153, 153, 153])
    cmap[6] = np.array([250, 170, 30])
    cmap[7] = np.array([220, 220, 0])
    cmap[8] = np.array([107, 142, 35])
    cmap[9] = np.array([152, 251, 152])
    cmap[10] = np.array([70, 130, 180])
    cmap[11] = np.array([220, 20, 60])
    cmap[12] = np.array([255,  0,  0])
    cmap[13] = np.array([0,  0, 142])
    cmap[14] = np.array([0,  0, 70])
    cmap[15] = np.array([0, 60, 100])
    cmap[16] = np.array([0, 80, 100])
    cmap[17] = np.array([0,  0, 230])
    cmap[18] = np.array([119, 11, 32])

    return cmap

cmap = color_map()

class SSDADataset(Dataset):
    def __init__(self, mode, labeled_num, root='/data/DataSets/'):
        self.root = root
        self.mode = mode
        self.labeled_num = labeled_num
        self.path = None
        self.size = 512
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        if self.mode == 'labeled':
            with open(self.root + 'Cityscapes/train.list', 'r') as f:
                self.path = f.read().splitlines()[:self.labeled_num]
            print('-----Total number of images in {} Cityscapes: {:d}'.format(mode, len(self.path)))
            GTAV_path = glob(self.root + 'GTAV/images/*.png')
            # print(GTAV_path)
            self.path = self.path + GTAV_path
            print('-----Total number of images in {} GTAV: {:d}'.format(mode, len(GTAV_path)))
        elif self.mode == 'unlabeled':
            with open(self.root + 'Cityscapes/train.list', 'r') as f:
                self.path = f.read().splitlines()[self.labeled_num:]
            print('-----Total number of images in {} Cityscapes: {:d}'.format(mode, len(self.path)))
        elif self.mode == 'test':
            with open(self.root + 'Cityscapes/val.list', 'r') as f:
                self.path = f.read().splitlines()
            print('-----Total number of images in {} Cityscapes: {:d}'.format(mode, len(self.path)))
    
    def __len__(self):
        return len(self.path)
    
    def __getitem__(self, item):
        id = self.path[item]
        if self.mode == 'test':
            img_path, mask_path = os.path.join(self.root, 'Cityscapes/', id.split(' ')[0]), os.path.join(self.root, 'Cityscapes/', id.split(' ')[1])
            img = Image.open(img_path).resize((self.size, self.size), Image.BILINEAR)
            mask = Image.open(mask_path).resize((self.size, self.size), Image.NEAREST)
            img, mask = normalize(img, mask)
            return img, mask, id
        else:
            if 'GTAV' in id:
                img = Image.open(id).resize((self.size, self.size), Image.BILINEAR)
                mask = Image.open(id.replace('images', 'labels')).resize((self.size, self.size), Image.NEAREST)
            else:
                img_path, mask_path = os.path.join(self.root, 'Cityscapes/', id.split(' ')[0]), os.path.join(self.root, 'Cityscapes/', id.split(' ')[1])
                img = Image.open(img_path).resize((self.size, self.size), Image.BILINEAR)
                mask = Image.open(mask_path).resize((self.size, self.size), Image.NEAREST)
            img, mask = resize(img, mask, (0.5, 2.0))
            img, mask = crop(img, mask, self.size)
            img, mask = hflip(img, mask, p=0.5)
            strong_img = img.copy()
            mask = np.array(mask)
            if 'GTAV' in id:
                label_copy = 255 * np.ones(mask.shape, dtype=np.float32)
                for k, v in self.id_to_trainid.items():
                    label_copy[mask == k] = v
                mask = label_copy
            img, mask = normalize(img, mask)
            if self.mode == 'labeled':
                return img, mask, id
            else:
                if random.random() < 0.8:
                    strong_img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(strong_img)
                strong_img = transforms.RandomGrayscale(p=0.2)(strong_img)
                strong_img = blur(strong_img, p=0.5)
                strong_img = normalize(strong_img)

                return img, strong_img, mask, id
            

# a = SSDADataset('labeled', 100)
# exit()

if __name__ == '__main__':
    import custom_transforms as tr
    from utils import decode_segmap
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt

    composed_transforms_tr = transforms.Compose([
        # tr.RandomHorizontalFlip(),
        # tr.RandomSized(512),
        tr.RandomRotate(15),
        tr.ToTensor()])

    voc_train = FundusSegmentation(#split='train1',
    splitid=[1,2],lb_domain=2,lb_ratio=0.2,
                                   transform=composed_transforms_tr)

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=0)
    print(len(dataloader))
    for ii, sample in enumerate(dataloader):
        # print(sample)
        exit(0)
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = tmp
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0]).astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

            break
    plt.show(block=True)

