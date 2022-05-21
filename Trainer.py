# -*- coding: utf-8 -*-
# @Time    : 8/16/2021 3:14 PM
# @Author  : YaoGengqi
# @FileName: Trainer.py
# @Software: PyCharm
# @Description: Training the model

'''
logging:
2021-08-24 : 加入_time() 和 _get_log()函数，用于记录时间和日志。
2021-08-26 : 修改了大量的if_cuda语句，直接使用to(self.device)代替。
2021-08-27 : 加入了SPSR网络模型，并用于多教师网络的训练。
2021-08-28 : 考虑损失函数的修改。
2021-09-01 : 放弃三教师网络的尝试，修改了边缘损失的计算对象，Lm = L1(Laplace(t1_pred), Laplace(sr_pred))
2021-09-01 : 是否尝试 MSE 损失 ？？ 否，差别不大
2021-09-06 : 截至目前的SPSR教师模型训练的RFDN效果在PI感知上效果很差。
2021-09-06 : 删除Teacher3
2021-09-18 : 是否考虑采用双通道的RFDN-D和特征图计算损失进行训练
2021-09-21 : 训练RFDN-L
2021-10-26 : 训练EdgeSRN和HR作为教师网络的RFDN
'''

import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from Datasets import DIV2K, Vaildation
from Utils import utils
from Utils.utils import get_folder
import skimage.color as sc
import random
from Utils.edge_detect import fraction_differential
from Model import *
import time, datetime

class Trainer():

    def __init__(self, args):

        self.args = args
        self._setting()

        self.model = get_model(model_name=args.model_name,
                               upscale=args.scale,
                               checkpoint=args.checkpoint).to(self.device)


        if self.args.Train == True:
            self._get_log()
            self._get_dataloader()


    def Test(self):

        for dataset in self.args.test_datasets:

            test_hr_folder, test_lr_folder, ext = get_folder(dataset, self.args.scale)
            filelist = utils.get_list(test_hr_folder, ext=ext)

            psnr_list = np.zeros(len(filelist))
            ssim_list = np.zeros(len(filelist))

            i = 0
            for imname in filelist:

                print("\r> Dataset: " + dataset + " [" + str(i) + "/" + str(len(filelist)) + "]", end="")
                # img operation
                im_gt = cv2.imread(imname, cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR to RGB
                im_gt = utils.modcrop(im_gt, self.args.scale)
                im_l = cv2.imread( test_lr_folder + imname.split('/')[-1].split('.')[0] + ext,
                    cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR to RGB

                if len(im_gt.shape) < 3:
                    im_gt = im_gt[..., np.newaxis]
                    im_gt = np.concatenate([im_gt] * 3, 2)
                    im_l = im_l[..., np.newaxis]
                    im_l = np.concatenate([im_l] * 3, 2)
                im_input = im_l / 255.0
                im_input = np.transpose(im_input, (2, 0, 1))
                im_input = im_input[np.newaxis, ...]
                im_input = torch.from_numpy(im_input).float()

                if self.cuda:
                    self.model = self.model.to(self.device)
                    im_input = im_input.to(self.device)

                with torch.no_grad():
                    out = self.model(im_input)
                    input_size = im_input.shape

                out_img = utils.tensor2np(out.detach()[0])

                crop_size = self.args.scale
                cropped_sr_img = utils.shave(out_img, crop_size)
                cropped_gt_img = utils.shave(im_gt, crop_size)
                if self.args.isY is True:
                    im_label = utils.quantize(sc.rgb2ycbcr(cropped_gt_img)[:, :, 0])
                    im_pre = utils.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])
                else:
                    im_label = cropped_gt_img
                    im_pre = cropped_sr_img

                psnr_list[i] = utils.compute_psnr(im_pre, im_label)
                ssim_list[i] = utils.compute_ssim(im_pre, im_label)

                # 存储输出图片
                if self.args.if_save_image:

                    output_path = self.args.output_folder + '/' \
                                  + self.args.model_name + '/' \
                                  + self.args.checkpoint[:-4] + '/'\
                                  + dataset + '/'

                    if not os.path.exists(output_path):
                        os.makedirs(output_path)

                    output_folder = os.path.join(output_path, imname.split('/')[-1].split('.')[0] + '.png')
                    cv2.imwrite(output_folder, out_img[:, :, [2, 1, 0]])

                i += 1

            print("\r===>  " + dataset + ":                   ")
            print("===>  PSNR: {}".format(np.mean(psnr_list)))
            print("===>  SSIM: {}".format(np.mean(ssim_list)))

    # 训练批次进行验证并输出指标数据，保存日志。
    def _valid(self):

        self.model.eval()
        avg_psnr, avg_ssim = 0, 0
        for i, batch in enumerate(self.test_loader, 1):
            lr_tensor, hr_tensor = batch[0].to(self.device), batch[1].to(self.device)

            with torch.no_grad():
                pre = self.model(lr_tensor)

            sr_img = utils.tensor2np(pre.detach()[0])
            gt_img = utils.tensor2np(hr_tensor.detach()[0])

            crop_size = self.args.scale
            cropped_sr_img = utils.shave(sr_img, crop_size)
            cropped_gt_img = utils.shave(gt_img, crop_size)

            if self.args.isY is True:
                im_label = utils.quantize(sc.rgb2ycbcr(cropped_gt_img)[:, :, 0])
                im_pre = utils.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])
            else:
                im_label = cropped_gt_img
                im_pre = cropped_sr_img

            avg_psnr += utils.compute_psnr(im_pre, im_label)
            avg_ssim += utils.compute_ssim(im_pre, im_label)

        avg_psnr = avg_psnr / len(self.test_loader)
        avg_ssim = avg_ssim / len(self.test_loader)

        if avg_ssim >= self.best_ssim and avg_psnr >= self.best_psnr:
            self.best_psnr = avg_psnr
            self.best_ssim = avg_ssim
            self._save_checkpoint()

        msg = "#### Valid({}): Epoch[{:03d}/{}]   ".format(self.args.valid_dataset, self.epoch, self.args.nEpochs)
        msg += "PSNR/Best_PSNR: {:.4f}/{:.4f}   SSIM/Best_SSIM: {:.4f}/{:.4f}   ".format(avg_psnr, self.best_psnr, avg_ssim, self.best_ssim)
        msg += "LR_init: {:.7f}   ".format(self.args.lr)


        print("\r                                                                                                                                  ", end="")
        print("\r"+msg, end="")
        with open(self.log_path, 'a') as self.log:
            print(str(datetime.datetime.now())[:19] + " INFO: " + msg[5:], file=self.log)

    # 存储权值文件
    def _save_checkpoint(self):

        torch.save(self.model.state_dict(), self.model_out_path)

        print("\r                                                                                                    ", end="")
        print("\r#### Checkpoint save in : " + self.model_out_path)

        with open(self.log_path, 'a') as self.log:
            print(str(datetime.datetime.now())[:19] + " INFO: Checkpoint save in : " + self.model_out_path, file=self.log)

    # 初始化设置
    def _setting(self):

        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)

        self.cuda = torch.cuda.is_available()

        if self.cuda:
            torch.backends.cudnn.benchmark = True
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        self.epoch = self.args.start_epoch
        self.train_by_TT = self.args.train_by_TT
        self.best_psnr = 20
        self.best_ssim = 0

        if self.args.Train == True:
            self.model_train_set = input("Please input the train_set to set the model name and log name : \n")

        # 创建文件夹
            model_folder = "Checkpoints/" + self.args.model_name + "/"
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)
            self.model_out_path = model_folder + \
                                  self.args.model_name + "_x" + str(self.args.scale) + "_" + self.model_train_set + ".pth"

    # 用于获取数据集
    def _get_dataloader(self):

        train_dataset = DIV2K.div2k(self.args)

        hr_dir = "Datasets/" + self.args.valid_dataset + "/"
        lr_dir = "Datasets/" + self.args.valid_dataset + "_LR/x" + str(self.args.scale) + "/"
        valid_dataset = Vaildation.DatasetFromFolderVal(hr_dir, lr_dir, self.args.scale)

        self.train_loader = DataLoader(dataset=train_dataset,
                                       num_workers=self.args.num_workers,
                                       batch_size=self.args.batch_size,
                                       shuffle=True,
                                       pin_memory=True,
                                       drop_last=True)

        self.test_loader = DataLoader(dataset=valid_dataset,
                                      num_workers=self.args.num_workers,
                                      batch_size=1,
                                      shuffle=False)

    # 用于计算训练所需的时长
    def _time(self, seconds):

        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)

        print("Epoch_Time:%02dh:%02dm  " % (h, m), end="")

        seconds = seconds * (self.args.nEpochs - self.epoch)
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        print("Rest_Time:%03dh%02dm" % (h, m))


    # 创建用于保存记录的log文件，并保存验证机的指标记录和checkpoint的保存记录。
    def _get_log(self):

        args = self.args
        log_name = str(datetime.datetime.now())[:11] + args.model_name + "_x" + str(args.scale) + "_"
        log_name += self.model_train_set

        self.log_path = "logs/" + log_name  + ".txt"
        self.train_loss_log_path = "logs/" + log_name + "_loss.txt"

        with open(self.log_path, 'w') as self.log:
             for key in list(vars(args).keys()):
                 print("Train_Args[%15s]:\t %s" % (key, vars(args)[key]), file=self.log)
        with open(self.train_loss_log_path, 'w') as self.log:
             print("loss ", file=self.log)
