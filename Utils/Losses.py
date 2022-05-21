# -*- coding: utf-8 -*-
# @Time    : 2021/4/23 14:41
# @Author  : YaoGengqi
# @FileName: Losses.py
# @Software: PyCharm
# @Description:

import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class VGGFeatureExtractor(nn.Module):
    # [N, 3, H, W] -> [N, 512, H/16, W/16]
    def __init__(self,
                 feature_layer=34,
                 use_bn=False,
                 use_input_norm=True,
                 device=torch.device('cpu')):

        super(VGGFeatureExtractor, self).__init__()
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        self.use_input_norm = use_input_norm

        if self.use_input_norm:

            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

            self.register_buffer('mean', mean)
            self.register_buffer('std', std)

        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output



