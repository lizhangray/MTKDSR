import numpy as np
import os
try:
    from skimage.measure import compare_psnr, compare_ssim
except:
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr
    from skimage.metrics import structural_similarity as compare_ssim

#  计算psnr值
def compute_psnr(im1, im2, crop_border=0):

    if crop_border > 0:
        im1 = shave(im1, crop_border)
        im2 = shave(im2, crop_border)

    return compare_psnr(im1, im2)


#  计算ssim值
def compute_ssim(im1, im2, crop_border=0):

    if crop_border > 0:
        im1 = shave(im1, crop_border)
        im2 = shave(im2, crop_border)

    isRGB = len(im1.shape) == 3 and im1.shape[-1] == 3
    s = compare_ssim(im1, im2, K1=0.01, K2=0.03, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
             multichannel=isRGB)
    return s

#  用于裁剪图像的边缘，在图像的检测指标之前。
def shave(im, border):
    border = [border, border]
    im = im[border[0]:-border[0], border[1]:-border[1], ...]
    return im


# 输入测试文件夹的名字， 返回HR和LRw文件夹的相对路径和图片格式
def get_folder(dataset, scale):
    hr_folder = 'Datasets/' + dataset + '/'
    lr_folder = 'Datasets/' + dataset + '_LR/x' + str(scale) + '/'

    return hr_folder, lr_folder, '.png'


# 获取path路径下以.ext文件格式保存的图片路径列表
def get_list(path, ext):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(ext)]

def modcrop(im, modulo):
    sz = im.shape
    h = np.int32(sz[0] / modulo) * modulo
    w = np.int32(sz[1] / modulo) * modulo
    ims = im[0:h, 0:w, ...]
    return ims


def convert_shape(img):
    img = np.transpose((img * 255.0).round(), (1, 2, 0))
    img = np.uint8(np.clip(img, 0, 255))
    return img

def quantize(img):
    return img.clip(0, 255).round().astype(np.uint8)

def tensor2np(tensor, out_type=np.uint8, min_max=(0, 1)):
    tensor = tensor.float().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0, 1]
    img_np = tensor.numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()

    return img_np.astype(out_type)

def convert2np(tensor):
    return tensor.cpu().mul(255).clamp(0, 255).byte().squeeze().permute(1, 2, 0).numpy()
