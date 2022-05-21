import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

from . block import *

class RFDN_L(nn.Module):
    def __init__(self, in_nc=3, nf=64, num_modules=6, out_nc=3, upscale=4):
        super(RFDN_L, self).__init__()

        self.fea_conv = conv_layer(in_nc, nf, kernel_size=3)  # 特征提取模块，C: 3 -> 64

        self.B1 = RFDB(in_channels=nf)
        self.B2 = RFDB(in_channels=nf)
        self.B3 = RFDB(in_channels=nf)
        self.B4 = RFDB(in_channels=nf)
        self.B5 = RFDB(in_channels=nf)
        self.B6 = RFDB(in_channels=nf)

        self.c = conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')    # 拼接模块， 4 * 64 -> 64
        self.LR_conv = conv_layer(nf, nf, kernel_size=3)
        self.upsampler = pixelshuffle_block(nf, out_nc, upscale_factor=upscale)
        self.scale_idx = 0

    def forward(self, input):
        out_fea = self.fea_conv(input)

        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B5 = self.B3(out_B4)
        out_B6 = self.B4(out_B5)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea
        output = self.upsampler(out_lr)

        return output

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx

def load_state_dict(path):

    if torch.cuda.is_available():
        state_dict = torch.load(path)
    else:
        state_dict = torch.load(path, map_location='cpu')

    new_state_dcit = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            name = k[7:]    # 去掉k中的’module.‘正好七个字符
        else:
            name = k
        new_state_dcit[name] = v
    return new_state_dcit

def get_RFDN_L(in_nc=3, out_nc=3, upscale=4, checkpoint=None):

    model = RFDN_L(upscale=upscale, in_nc=in_nc, out_nc=out_nc)
    if checkpoint is not None:
        checkpoint = './Checkpoints/RFDN_L/' + checkpoint
        model_dict = load_state_dict(checkpoint)
        model.load_state_dict(model_dict, strict=True)

    return model