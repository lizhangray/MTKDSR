# -*- coding: utf-8 -*-
# @Time    : 2021/4/23 11:57
# @Author  : YaoGengqi
# @FileName: cityscapes.py
# @Software: PyCharm
# @Description:


try:
    from .base_datasets import BaseDataset
except:
    from base_datasets import BaseDataset
import numpy as np
import torch
import random
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T

##############################################################################
labels_info = [
    {"hasInstances": False, "category": "void",         "catid": 0, "name": "unlabeled",            "ignoreInEval": True,  "id": 0,  "color": [0, 0, 0],        "trainId": 255},
    {"hasInstances": False, "category": "void",         "catid": 0, "name": "ego vehicle",          "ignoreInEval": True,  "id": 1,  "color": [0, 0, 0],        "trainId": 255},
    {"hasInstances": False, "category": "void",         "catid": 0, "name": "rectification border", "ignoreInEval": True,  "id": 2,  "color": [0, 0, 0],        "trainId": 255},
    {"hasInstances": False, "category": "void",         "catid": 0, "name": "out of roi",           "ignoreInEval": True,  "id": 3,  "color": [0, 0, 0],        "trainId": 255},
    {"hasInstances": False, "category": "void",         "catid": 0, "name": "static",               "ignoreInEval": True,  "id": 4,  "color": [0, 0, 0],        "trainId": 255},
    {"hasInstances": False, "category": "void",         "catid": 0, "name": "dynamic",              "ignoreInEval": True,  "id": 5,  "color": [111, 74, 0],     "trainId": 255},
    {"hasInstances": False, "category": "void",         "catid": 0, "name": "ground",               "ignoreInEval": True,  "id": 6,  "color": [81, 0, 81],      "trainId": 255},
    {"hasInstances": False, "category": "flat",         "catid": 1, "name": "road",                 "ignoreInEval": False, "id": 7,  "color": [128, 64, 128],   "trainId": 0},
    {"hasInstances": False, "category": "flat",         "catid": 1, "name": "sidewalk",             "ignoreInEval": False, "id": 8,  "color": [244, 35, 232],   "trainId": 1},
    {"hasInstances": False, "category": "flat",         "catid": 1, "name": "parking",              "ignoreInEval": True,  "id": 9,  "color": [250, 170, 160],  "trainId": 255},
    {"hasInstances": False, "category": "flat",         "catid": 1, "name": "rail track",           "ignoreInEval": True,  "id": 10, "color": [230, 150, 140],  "trainId": 255},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "building",             "ignoreInEval": False, "id": 11, "color": [70, 70, 70],     "trainId": 2},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "wall",                 "ignoreInEval": False, "id": 12, "color": [102, 102, 156],  "trainId": 3},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "fence",                "ignoreInEval": False, "id": 13, "color": [190, 153, 153],  "trainId": 4},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "guard rail",           "ignoreInEval": True,  "id": 14, "color": [180, 165, 180],  "trainId": 255},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "bridge",               "ignoreInEval": True,  "id": 15, "color": [150, 100, 100],  "trainId": 255},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "tunnel",               "ignoreInEval": True,  "id": 16, "color": [150, 120, 90],   "trainId": 255},
    {"hasInstances": False, "category": "object",       "catid": 3, "name": "pole",                 "ignoreInEval": False, "id": 17, "color": [153, 153, 153],  "trainId": 5},
    {"hasInstances": False, "category": "object",       "catid": 3, "name": "polegroup",            "ignoreInEval": True,  "id": 18, "color": [153, 153, 153],  "trainId": 255},
    {"hasInstances": False, "category": "object",       "catid": 3, "name": "traffic light",        "ignoreInEval": False, "id": 19, "color": [250, 170, 30],   "trainId": 6},
    {"hasInstances": False, "category": "object",       "catid": 3, "name": "traffic sign",         "ignoreInEval": False, "id": 20, "color": [220, 220, 0],    "trainId": 7},
    {"hasInstances": False, "category": "nature",       "catid": 4, "name": "vegetation",           "ignoreInEval": False, "id": 21, "color": [107, 142, 35],   "trainId": 8},
    {"hasInstances": False, "category": "nature",       "catid": 4, "name": "terrain",              "ignoreInEval": False, "id": 22, "color": [152, 251, 152],  "trainId": 9},
    {"hasInstances": False, "category": "sky",          "catid": 5, "name": "sky",                  "ignoreInEval": False, "id": 23, "color": [70, 130, 180],   "trainId": 10},
    {"hasInstances": True,  "category": "human",        "catid": 6, "name": "person",               "ignoreInEval": False, "id": 24, "color": [220, 20, 60],    "trainId": 11},
    {"hasInstances": True,  "category": "human",        "catid": 6, "name": "rider",                "ignoreInEval": False, "id": 25, "color": [255, 0, 0],      "trainId": 12},
    {"hasInstances": True,  "category": "vehicle",      "catid": 7, "name": "car",                  "ignoreInEval": False, "id": 26, "color": [0, 0, 142],      "trainId": 13},
    {"hasInstances": True,  "category": "vehicle",      "catid": 7, "name": "truck",                "ignoreInEval": False, "id": 27, "color": [0, 0, 70],       "trainId": 14},
    {"hasInstances": True,  "category": "vehicle",      "catid": 7, "name": "bus",                  "ignoreInEval": False, "id": 28, "color": [0, 60, 100],     "trainId": 15},
    {"hasInstances": True,  "category": "vehicle",      "catid": 7, "name": "caravan",              "ignoreInEval": True,  "id": 29, "color": [0, 0, 90],       "trainId": 255},
    {"hasInstances": True,  "category": "vehicle",      "catid": 7, "name": "trailer",              "ignoreInEval": True,  "id": 30, "color": [0, 0, 110],      "trainId": 255},
    {"hasInstances": True,  "category": "vehicle",      "catid": 7, "name": "train",                "ignoreInEval": False, "id": 31, "color": [0, 80, 100],     "trainId": 16},
    {"hasInstances": True,  "category": "vehicle",      "catid": 7, "name": "motorcycle",           "ignoreInEval": False, "id": 32, "color": [0, 0, 230],      "trainId": 17},
    {"hasInstances": True,  "category": "vehicle",      "catid": 7, "name": "bicycle",              "ignoreInEval": False, "id": 33, "color": [119, 11, 32],    "trainId": 18},
    {"hasInstances": False, "category": "vehicle",      "catid": 7, "name": "license plate",        "ignoreInEval": True,  "id": -1, "color": [0, 0, 142],      "trainId": -1}
]
color_map = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
for el in labels_info:
    color_map[el['trainId']] = el["color"]
###############################################################################

class Cityscapes(BaseDataset):

    def __init__(self,
                 dataroot='../datasets/cityscapes/',
                 txt_path='../datasets/cityscapes/train.txt',
                 transforms=True,
                 crop_size=(1024, 1024),
                 train=True):
        super(Cityscapes, self).__init__(dataroot=dataroot, txt_path=txt_path)

        self.n_classes = 19
        self.ignore_label = 255

        self.transforms = transforms
        self.train = train
        self.size = crop_size

        self.to_tensor = T.ToTensor()
        self.to_PIL = T.ToPILImage()

        self.flip = T.RandomHorizontalFlip(p=0.5)
        self.resize = T.RandomResizedCrop(self.size)
        self.lb_map = np.arange(256).astype(np.uint8)

        for el in labels_info:
            self.lb_map[el['id']] = el['trainId']

    def __getitem__(self, idx):

        label = Image.open(self.label_paths[idx])
        if self.transforms:
            label = self._transforms(label)

        return self._down_sample(label), self._down_sample(self._down_sample(label))

    def _transforms(self, label):

        label = self.to_tensor(label) # (1， 1024， 2048)

        concat = torch.cat([label, label], 0)
        concat = self.resize(self.to_PIL(concat))
        if self.train:
            self.flip(concat)

        concat = self.to_tensor(concat)
        label = concat[1:, :, :]

        return self._trans_id(label)

    def _trans_id(self, label):

        label = np.array(self.to_PIL(label))
        label = self.lb_map[label]
        return torch.from_numpy(label)

    def _down_sample(self, HR):

        if len(HR.shape) == 3:
            HR = HR.unsqueeze(1).float()
        elif len(HR.shape) == 2:
            HR = HR.unsqueeze(0).unsqueeze(0).float()

        LR = torch.nn.functional.interpolate(HR, scale_factor=0.5) # , mode='bicubic')

        return torch.clamp(LR.squeeze().int(), min=0, max=255)


def get_dataLoader( dataroot='..../Project/torch_seg/datasets/cityscapes/',
                    txt_path='..../Project/torch_seg/datasets/cityscapes/train.txt',
                    transforms=True,
                    crop_size=(1024, 1024),
                    train=False,
                    batch_size = 8,
                    num_workers = 0,
                    pin_memory=False):

    return DataLoader(Cityscapes(dataroot, txt_path, transforms, crop_size, train),
                      batch_size=batch_size,
                      shuffle=train,
                      num_workers=num_workers,
                      pin_memory=pin_memory,
                      drop_last=True)

def show_label(label):

    color_map = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
    for el in labels_info:
        color_map[el['trainId']] = el["color"]

    label = np.array(label)
    label = Image.fromarray(color_map[label])
    Image._show(label)

def save_label(label, epoch, name='default'):

    label = torch.clamp(label, min=0, max=255)

    color_map = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
    for el in labels_info:
        color_map[el['trainId']] = el["color"]

    label = np.array(label.cpu())
    label = Image.fromarray(color_map[label])
    label.save('/mnt/nvme1n1p1/ygq/Files/OneDrive/Project/TT_SR/Result/Cityscape/Epoch_' + str(epoch) + '_' + name + '.png')

if __name__ == "__main__":

    ds = Cityscapes(dataroot='C:/Users/admin/OneDrive/Project/torch_seg/datasets/cityscapes/',
                    txt_path='C:/Users/admin/OneDrive/Project/torch_seg/datasets/cityscapes/train.txt')
    img, label = ds.__getitem__(2)
    show_label(img)
    show_label(label)
    #torch.Size([512, 512]) torch.Size([256, 256])

    dl = get_dataLoader(dataroot='C:/Users/admin/OneDrive/Project/torch_seg/datasets/cityscapes/',
                        txt_path='C:/Users/admin/OneDrive/Project/torch_seg/datasets/cityscapes/train.txt',)
    for i, (imgs, label) in enumerate(dl):
        print(imgs.shape, label.shape)
        break


