# -*- coding: utf-8 -*-
# @Time    : 2021/10/18 13:40
# @Author  : YaoGengqi
# @FileName: down.py
# @Software: PyCharm
# @Description:


from os import listdir
from os.path import join
from torchvision.transforms import Compose,  CenterCrop, Resize
from PIL import Image
import os

def is_imagefile(image):
    return any(image.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG',
                                                               '.JPEG','bmp','BMP'])

def calculate_valid_crop_size(crop_size,upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def hr_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
    ])

def lr_transform(crop_size):
    return Compose([
        Resize(crop_size, interpolation=Image.BICUBIC),
    ])


def produce_image(data_dir,scale):

    filename = [join(data_dir, x) for x in listdir(data_dir) if is_imagefile(x)]

    for x in filename:
        images_name = x.split('/')[-1]
        images_name = images_name.split('.')[0]
        x_image = Image.open(x)
        (w,h) = x_image.size
        print(w,h)
        nw = calculate_valid_crop_size(w,24)
        nh = calculate_valid_crop_size(h,24)
        hr_size = hr_transform((nh,nw))
        x_image = hr_size(x_image)
        print(x_image)
        save_image(x_image,scale,images_name)

def save_image(x_image,scale,images_name):
    output_lr_dir = 'Datasets/Set5_LR/x4_ss'
    output_hr_dir = 'Datasets/Set5'

    x_image.save(os.path.join(output_hr_dir,images_name + '.bmp'))
    for s in scale:
        os.makedirs(
            os.path.join(output_lr_dir,'X{}'.format(s)),
            exist_ok= True
        )
        path = os.path.join(output_lr_dir,'X{}'.format(s) + '/' + images_name + '_X{}'.format(s) + '.bmp')
        (nw,nh) = x_image.size
        lr_size = lr_transform((nh // s, nw // s))
        xr_image = lr_size(x_image)
        xr_image.save(path)




data_dir = 'Datasets/Set5'
scale = [4]
produce_image(data_dir,scale)