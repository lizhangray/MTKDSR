# -*- coding: utf-8 -*-
# @Time    : 2021/4/23 11:35
# @Author  : YaoGengqi
# @FileName: base_datasets.py
# @Software: PyCharm
# @Description:

import os
from PIL import Image
from torch.utils.data import Dataset

class BaseDataset(Dataset):

    def __init__(self,
                 dataroot,      # 指代数据集的根目录，如 '.datasets/cityspaces'
                 txt_path ):    # 指代记录数据集的txt文件，如cityspaces的train.txt，拼接root后便是完整路径

        super(BaseDataset, self).__init__()

        self.root = dataroot
        self.txt = txt_path
        self.img_paths, self.label_paths = self._get_path()

    def __getitem__(self, idx):

        img_path, gt_path = self.img_paths[idx], self.label_paths[idx]

        img = Image.open(img_path).convert('RGB')   # (H, W, 3)
        label = Image.open(gt_path)                 # (H, W)

        return img, label


    def __len__(self):
        return len(self.img_paths)

    def _get_path(self):  # 加载img和label的完整路径

        img_paths, label_paths = [], []

        with open(self.txt, 'r') as f:
            img_gt_pairs = f.read().splitlines()

        for pair in img_gt_pairs:
            img_path, gt_path = pair.split(',')

            img_paths.append(os.path.join(self.root, img_path))
            label_paths.append(os.path.join(self.root, gt_path))

        assert len(img_paths) == len(label_paths)

        return img_paths, label_paths

