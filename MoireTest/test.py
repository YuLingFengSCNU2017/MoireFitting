# school library test
import MoireTest.functions_1 as f1
import MoireTest.openCVfft as ocf
import numpy as np
import cv2

a = f1.gg('1.jpg')
a = cv2.equalizeHist(a)
a = a / 225
cv2.imshow('origin!!', a)
b = np.fft.fft2(a)
b = np.fft.fftshift(b)
b = np.abs(b)
ocf.logplt(b)
cv2.waitKey(0)
cv2.destroyAllWindows()
