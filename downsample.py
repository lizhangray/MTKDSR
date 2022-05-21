# -*- coding: utf-8 -*-
# @Time    : 2021/10/17 14:53
# @Author  : YaoGengqi
# @FileName: downsample.py
# @Software: PyCharm
# @Description:

import os
from PIL import Image
from torchvision.transforms import Compose, CenterCrop, Resize

def hr_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
    ])

def lr_transform(crop_size):
    return Compose([
        Resize(crop_size, interpolation=Image.BICUBIC),
    ])

def down_bicubic(data_root):
    """将文件夹内的图片进行下采样并保存"""

    hr_output = data_root
    lr_output = data_root + r'_LR\x4'  # 输出路径

    if not os.path.exists(lr_output):
        os.mkdir(lr_output)
    dirs = os.listdir(data_root)

    i = 0
    for file in dirs:

        i += 1
        if not file.endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', 'bmp', 'BMP')):
            continue

        # 存储并修改hr_img边长为4的倍数
        hr_img = Image.open(os.path.join(data_root, file))
        w, h = hr_img.size
        crop_h, crop_w = h-(h%24), w-(w%24)
        hr_size = hr_transform((crop_h, crop_w))
        hr_img = hr_size(hr_img)
        hr_img.save(os.path.join(hr_output, file[:-4] + '.png'))

        # 下采样4倍数并存储
        lr_size = lr_transform((crop_h//4, crop_w//4))
        lr_img = lr_size(hr_img)
        lr_img.save(os.path.join(lr_output, file[:-4] + '.png'))

        print("\rSaving [" + str(i) + "/" + str(len(dirs)) + '] : ' + os.path.join(lr_output, file), end="")

def jpg2png(data_root):

    dirs = os.listdir(data_root)

    i = 0
    for file in dirs:

        i += 1
        if not file.endswith(('.jpg')):
            continue

        jpg_img = Image.open(os.path.join(data_root, file))
        file = os.path.splitext(file)[0] + '.png'
        jpg_img.save(os.path.join(data_root, file))

        print("\rSaving [" + str(i) + "/" + str(len(dirs)) + '] : ' + os.path.join(data_root, file), end="")

    return

def upsample_bicubic(lr_path, hr_output):

    if not os.path.exists(hr_output):
        os.mkdir(hr_output)
    dirs = os.listdir(lr_path)

    i = 0
    for file in dirs:

        i += 1
        if not file.endswith('.png'):
            continue

        # 存储并修改hr_img边长为4的倍数
        lr_img = Image.open(os.path.join(lr_path, file))
        w, h = lr_img.size

        hr_size = lr_transform((h*4, w*4))
        hr_img = hr_size(lr_img)
        hr_img.save(os.path.join(hr_output, file[:-4] + '.png'))

        print("\rSaving [" + str(i) + "/" + str(len(dirs)) + '] : ' + os.path.join(hr_output, file), end="      ")


# down_bicubic(data_root=r'D:\OneDrive\Project\TT_SR\Datasets\Set14')

# jpg2png(r'D:\OneDrive\Project\TT_SR\Datasets\BSD100')

upsample_bicubic(r'D:\OneDrive\Project\TT_SR\Datasets\Manga109_LR\x4', r'D:\OneDrive\Project\TT_SR\Result\BICUBIC\Manga109')