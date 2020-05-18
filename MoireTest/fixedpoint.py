import numpy as np
import cv2


def fimg(im):
    imf = np.fft.fft2(im)
    imfs = np.fft.fftshift(imf)
    normnum = 1 / np.max(np.abs(imfs))
    imfs = imfs * normnum
    return (imfs + im) / 2


pix, L = 80, 0.001
im0_1 = np.random.random([pix, pix])
imx = np.linspace(-1, 1, pix)
imx, imy = np.meshgrid(imx, imx)

im0_2 = (1 / (L * np.sqrt(2 * np.pi)))*np.exp(-(imx ** 2 + imy ** 2) / (2 * L ** 2))
im0 = np.abs(im0_1 * 0.2 + im0_2 * 0.8)
cv2.imwrite('1.jpg', (im0 * 255).astype(int))
cv2.imshow('img', im0)
print('@@@1\n',im0)
cv2.waitKey(0)
im0 = np.fft.fft2(im0)
im0 = np.abs(np.fft.fftshift(im0))
im0 = im0 / np.max(im0)
cv2.imshow('img', np.abs(im0))
print('@@@2\n',im0)
cv2.waitKey(0)
for i in range(10):
    im0 = fimg(im0)
cv2.imshow('img', np.abs(im0))
print('@@@3\n',im0)
cv2.waitKey(0)
