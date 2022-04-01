import torch, torchvision
from torch.utils import data
from torchvision import transforms

import math,random,os,scipy
import numpy as np
from PIL import Image

from utils_data import *


#########################################################################
# Images TRAINING SETTINGS
#########################################################################
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485 , 0.456 , 0.406] , [0.229 , 0.224 , 0.225])
])

map_transform = transforms.Compose([
    transforms.ToTensor()
])

fix_transform = transforms.Compose([
    transforms.ToTensor()
])

class TestData(data.Dataset):
    def __init__(self, root, img_transform=img_transform):
        self.root = os.path.expanduser(root)
        self.img_transform = img_transform

        imgs_path = os.path.join(self.root)
        self.imgs_list = [imgs_path + f for f in os.listdir(imgs_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.imgs_list.sort()

    def __getitem__(self, index):

        img_path = self.imgs_list[index]
        img = np.array(Image.open(img_path).convert('RGB'))
        img_name = os.path.split(img_path)[1][:-4]
        img_size = (img.shape[1],img.shape[0])

        if self.img_transform is not None:
            img = self.img_transform(img)

        return img, img_name, img_size


    def __len__(self):

        return len(self.imgs_list)

    def get_datasize(self):
        return len(self.imgs_list)

def test_loader(datapath, iosize=[480,640,60,80], batch_size=4, num_workers=0):

    input_h, input_w, target_h, target_w = iosize
    img_transform = transforms.Compose([
        transforms.Lambda(lambda x: padding(x, shape_r=input_h, shape_c=input_w)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = TestData(root=datapath, img_transform=img_transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return dataloader


class SALICON(data.Dataset):
    def __init__(self, root, classes='train',
                 img_transform=img_transform, map_transform=map_transform, fix_transform=fix_transform):
        self.root = os.path.expanduser(root)
        self.img_transform = img_transform
        self.map_transform = map_transform
        self.fix_transform = fix_transform

        # dset_opts = ['train', 'val', 'test']
        self.classes = classes

        imgs_path = os.path.join(self.root, self.classes, 'images/')
        self.imgs_list = [imgs_path + f for f in os.listdir(imgs_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.imgs_list.sort()

        if self.classes == 'test':
            self.maps_list = []
            self.fixs_list = []
        else:
            maps_path = os.path.join(self.root, self.classes, 'maps/')
            fixs_path = os.path.join(self.root, self.classes, 'fixations', 'maps/')

            self.maps_list = [maps_path + f for f in os.listdir(maps_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            self.fixs_list = [fixs_path + f for f in os.listdir(fixs_path) if f.endswith('.mat')]

            self.maps_list.sort()
            self.fixs_list.sort()

    def __getitem__(self, index):

        img_path = self.imgs_list[index]
        img = Image.open(img_path).convert('RGB')

        img_name = os.path.split(img_path)[1][:-4]
        img_size = (img.size[1],img.size[0])

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.classes == 'test':
            return img, img_name, img_size
        else:
            map_path = self.maps_list[index]
            map = Image.open(map_path).convert('L')

            fix_path = self.fixs_list[index]
            fix = scipy.io.loadmat(fix_path)["I"]

            if self.map_transform is not None:
                map = self.map_transform(map)

            if self.fix_transform is not None:
                fix = self.fix_transform(fix)

            return img, map, fix, img_name, img_size

    def __len__(self):

        return len(self.imgs_list)

    def get_datasize(self):
        return len(self.imgs_list)

def salicon_loader(datapath, classes='train', iosize=[480,640,60,80], batch_size=4, num_workers=0):

    input_h, input_w, target_h, target_w = iosize
    img_transform = transforms.Compose([
        transforms.Resize((input_h,input_w)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    map_transform = transforms.Compose([
        transforms.Resize((target_h,target_w)),
        transforms.ToTensor()
    ])
    fix_transform = transforms.Compose([
        transforms.Lambda(lambda x: padding_fixation(x, shape_r=target_h, shape_c=target_w)),
        transforms.Lambda(lambda x: np.expand_dims(x,axis=2)),
        transforms.ToTensor()
    ])

    if classes == 'train':
        shuffle = True
    else:
        shuffle = False
    dataset = SALICON(root=datapath, classes=classes, img_transform=img_transform, map_transform=map_transform, fix_transform=fix_transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader
