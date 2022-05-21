import numpy as np
import cv2
from PIL import Image
import scipy.signal as signal     # 导入sicpy的signal模块
import math
from PIL import Image
import torchvision.transforms as transforms


class fraction_differential(object):
    """docstring for fraction_differential"""
    def __init__(self, v=0, althea=1):
        a1 = - v
        a2 = (-v)*(-v+1)/2
        a3 = (-v)*(-v+1)*(-v+2)/6
        self.transform = transforms.Compose([#transforms.Normalize(mean = [-2.118, -2.036, -1.804], # Equivalent to un-normalizing ImageNet (for correct visualization)
                                             #                       std = [4.367, 4.464, 4.444]),
                                            transforms.ToPILImage(),
                                            ])
        self.althea = althea
        # fraction_differential算子
        self.suanzi1 = np.array([[a1, a1, a1],  
                                [a1, 8 , a1],
                                [a1, a1, a1]])

        # fraction_differential扩展算子
        self.suanzi2 = np.array([[a2, 0,  a2, 0, a2],
                                [0 , a1, a1, a1, 0],
                                [a2, a1, 8 , a1, a2],
                                [0 , a1, a1, a1, 0],
                                [a2, 0,  a2, 0, a2]])

        self.suanzi3 = np.array([[a3, 0,  0, a3, 0, 0, a3],
                                [0 , a2, 0,  a2, 0, a2,0],
                                [0 ,0 , a1, a1, a1, 0 ,0],
                                [a3, a2, a1, 8 , a1, a2,a3],
                                [0 ,0 , a1, a1, a1, 0, 0],
                                [0 , a2, 0,  a2, 0, a2,0],
                                [a3, 0,  0, a3, 0, 0, a3]])
        self.sobelx = np.array([[-1, 0, 1],  
                                [-2, 0 , 2],
                                [-1, 0, 1]])
        self.sobely = np.array([[1, 2, 1],  
                                [0, 0 , 0],
                                [-1, -2, -1]])
        self.sobell = np.array([[2, 1, 0],  
                                [1, 0 , -1],
                                [0, -1, -2]])
        self.sobelr = np.array([[0, 1, 2],  
                                [-1, 0 , 1],
                                [-2, -1, 0]])
        self.laplace = np.array([[0, 1, 0],  
                                [1, -4, 1],
                                [0, 1, 0]])
        self.robertsx = np.array([[-1,0],
                                  [0, 1]])
        self.robertsy = np.array([[0,-1],
                                  [1,0]])

    # def fraction_order(self, image1):
    #     image_array1 = np.array(image1)
    #     return output

    def roberts(self,image1):
        outputs = np.zeros((16,1,192,192))
        for i in range(image1.shape[0]):
            image2 = self.transform(image1[i])
            r,g,b = image2.split()
            r = np.array(r)
            g = np.array(g)
            b = np.array(b)
            image_array1 = 0.256789*r + 0.504129*g + 0.097906*b + 16
            image_robertsx = signal.convolve2d(image_array1,self.robertsx,mode="same")
            image_robertsy = signal.convolve2d(image_array1,self.robertsy,mode="same")
            image_roberts  = (image_robertsx**2+image_robertsy**2)**0.5 ##
            
            max_v = float(np.abs(image_roberts).max())
            outputs[i] = (image_roberts / max_v)
        return outputs

    def sobel(self,image1):
        batch ,w= image1.shape[0],image1.shape[2]
        if w == 32:
            outputs = np.zeros((batch,1,32,32))
        else:
            outputs = np.zeros((batch,1,192,192))
        for i in range(batch):
            image2 = self.transform(image1[i])
            r,g,b = image2.split()
            r = np.array(r)
            g = np.array(g)
            b = np.array(b)
            image_array1 = 0.256789*r + 0.504129*g + 0.097906*b + 16
            image_sobelx = signal.convolve2d(image_array1,self.sobelx,mode="same")
            image_sobely = signal.convolve2d(image_array1,self.sobely,mode="same")
            #image_sobell = signal.convolve2d(image_array1,self.sobell,mode="same")
            #image_sobelr = signal.convolve2d(image_array1,self.sobelr,mode="same")
            image_sobel  = (image_sobelx**2+image_sobely**2)**0.5 ##+image_sobell**2+image_sobelr**2
            
            max_v = image_sobel.max()
            output = (image_sobel / max_v)
            output[output *255.0 < 5] = 0
            outputs[i] = output
        #output = np.round(image_sobel).astype(np.uint8)
        #image_sobel = Image.fromarray(image_sobel)

        return outputs

    def Laplace(self,image1):
        batch = image1.shape[0]
        outputs = np.zeros((batch,1,192,192))
        for i in range(batch):
            image2 = self.transform(image1[i])
            r,g,b = image2.split()
            r = np.array(r)
            g = np.array(g)
            b = np.array(b)
            image_array1 = 0.256789*r + 0.504129*g + 0.097906*b + 16        
            image_laplace = signal.convolve2d(image_array1,self.laplace,mode="same")
            image_laplace = image_laplace.astype(np.float32)
            shift = np.abs(image_laplace.min())

            image_laplace = image_laplace + shift
            image_laplace = (image_laplace / image_laplace.max())

            outputs[i] = image_laplace

        return outputs  ##控制范围[0,1]

    def canny(self,image1):
        batch = image1.shape[0]
        outputs = np.zeros((batch,1,192,192))
        for i in range(batch):
            image2 = self.transform(image1[i])
            r,g,b = image2.split()
            r = np.array(r)
            g = np.array(g)
            b = np.array(b)
            image_array1 = 0.256789*r + 0.504129*g + 0.097906*b + 16
            #image_array1 = np.array(image1)
            ##-------------------canny-------------------
            #image = cv2.imread("./0010.png")#读入图像
            #image = cv2.cvtColor(image,cv2.COLOR_RGB2YCrCb)#将图像转化为灰度图像
            image_array1 = np.round(image_array1).astype(np.uint8)
            #print(image_array1)
            #while 1:
            #    pass
            #cv2.imshow("Image",image_array1)#显示图像
            #cv2.waitKey()
            #Canny边缘检测
            outputs[i] = cv2.Canny(image_array1,10,20)

            #np.set_printoptions(threshold=np.inf)
            #print(canny)
            #cv2.imshow("Canny",outputs[3])
            #cv2.waitKey()
        return outputs

    def Fourier(self,image1):
        batch = image1.shape[0]
        dftshift = np.zeros((batch,192,192,2))
        for i in range(batch):
            image2 = self.transform(image1[i])
            r,g,b = image2.split()
            r = np.array(r)
            g = np.array(g)
            b = np.array(b)
            image_array1 = 0.256789*r + 0.504129*g + 0.097906*b + 16

            #傅里叶变换
            dft = cv2.dft(np.float32(image_array1), flags = cv2.DFT_COMPLEX_OUTPUT)
            dftshift[i] = np.fft.fftshift(dft)
            #outputs[i]= 20*np.log(cv2.magnitude(dftshift[:,:,0], dftshift[:,:,1]))

        return dftshift


"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# 显示图像
plt.subplot(2,1,1)
plt.imshow(image_array1,cmap=cm.gray)
plt.axis("off")
plt.subplot(2,1,2)
plt.imshow(image_array2,cmap=cm.gray)
plt.axis("off")
plt.subplot(2,2,3)
plt.imshow(image_suanzi1,cmap=cm.gray)
plt.axis("off")
plt.subplot(2,2,4)
plt.imshow(image_suanzi2,cmap=cm.gray)
plt.axis("off")
plt.show()
"""