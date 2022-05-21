import torch
import glob
import cv2
import PIL.Image as Image
import torchvision

images = glob.glob("BSD100/*")
Totensor = torchvision.transforms.ToTensor()
ToImage  = torchvision.transforms.ToPILImage()

for img in images:
    print(img)
    img = Image.open(img)
    img = Totensor(img).unsqueeze(0)
    img = torch.nn.functional.interpolate(img, scale_factor=0.25, mode='bicubic').clamp(min=0, max=255)
    img = ToImage(img.squeeze())
    img.show()
    break

# LR = torch.nn.functional.interpolate(HR, scale_factor=0.5) # , mode='bicubic')