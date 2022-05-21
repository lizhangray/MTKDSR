# -*- coding: utf-8 -*-
# @Time    : 7/15/2021 3:52 PM
# @Author  : YaoGengqi
# @FileName: __init__.py
# @Software: PyCharm
# @Description:

from .IMDN import get_IMDN
from .RFDN import get_RFDN
from .block import VGGFeatureExtractor as get_Extractor
from .EdgeSRN import get_EdgeSRN
from .HAN import get_HAN
from .SPSR import get_SPSR, get_EdgeSPSR
from .ESRGAN import get_ESRGAN

def get_model(model_name, checkpoint, upscale):

    if model_name == 'IMDN':
        return get_IMDN(upscale=upscale, checkpoint=checkpoint)

    elif model_name == 'RFDN':
        return get_RFDN(upscale=upscale, checkpoint=checkpoint)

    elif model_name == 'EdgeSRN':
        return get_EdgeSRN(checkpoint=checkpoint)

    elif model_name == 'SPSR':
        return get_SPSR(checkpoint=checkpoint)

    elif model_name == 'EdgeSPSR':
        return get_EdgeSPSR(checkpoint=checkpoint)

    elif model_name == 'HAN':
        return get_HAN(upscale=upscale, checkpoint=checkpoint)

    elif model_name == 'ESRGAN':
        return get_ESRGAN(checkpoint=checkpoint)