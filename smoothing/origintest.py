import numpy as np
import cv2
from math import ceil


def filterpass(pq):  # 空间滤波器
    return pq


linewidth = 3
linespan = 15
[heig, leng] = [200, 200]
f = np.zeros([heig, leng])
for i in range(linewidth):
    a = np.arange(i, leng, linespan)
    f[:, a] = 1
for i in range(linewidth):
    a = np.arange(i, heig, linespan)
    f[a, :] = 1


pq = 2 ** np.array([ceil(np.log2(2 * heig - 1)), ceil(np.log2(2 * leng - 1))])
F = np.fft.fft2(f, pq)
F=np.fft.fftshift(F)
F=np.abs(F)/np.max(np.abs(F))
cv2.imshow('origin raster', F)
cv2.waitKey(0)
#H = filterpass(pq)
#G=np.fft.ifft2(H*F)
