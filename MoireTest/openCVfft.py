'''=========================================================================================
Date   :  2020.04.22
Author :  玩具反斗星
WeChat :  SCNU20172321114
Note   :  图像的傅里叶变换
========================================================================================='''

'''引用库和读取图片'''
import numpy as np
import cv2

'''======================================================================================'''

'''对数绘图'''


def logplt(im0, getlog=1):
    im0 = np.abs(im0)
    f0m = np.max(np.log(1 + np.abs(im0)))
    if getlog == 1:
        im0plt = np.log(1 + im0) / f0m  # np.max(np.log(im0))
    else:
        im0plt = im0
    cv2.imshow('image', im0plt)
    cv2.waitKey(0)


def ifftlog(image):
    f0fft = np.fft.ifftshift(image)
    im = np.abs(np.fft.ifft2(f0fft))
    return im / np.max(im)

def main():
    img = cv2.imread('1.jpg')

    imgray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 转化为灰度图

    '''傅里叶变换'''
    f = np.fft.fft2(imgray)
    f0 = np.fft.fftshift(f)  # 低频在中心
    '''绘制图像'''
    logplt(f0)

    '''滤波'''
    percent = 0.55  # 滤波保留中间的多少
    img0 = f0.copy()
    imgabs = np.abs(img0)
    img0 = img0 * imgabs * imgabs * imgabs
    middle1, middle2, del1, del2 = [round(img.shape[0] / 2), round(img.shape[1] / 2),
                                    round(img.shape[0] * percent / 2), round(img.shape[1] * percent / 2)]
    img0[middle1 - del1:middle1 + del1, middle2 - del2:middle2 + del2] = 0
    logplt(img0)

    img1 = f0 - img0
    logplt(img1)

    '''傅里叶逆变换'''
    img0 = ifftlog(img0)
    logplt(img0, 0)
    img1 = ifftlog(img1)
    logplt(img1, 0)
    logplt(imgray, 0)
