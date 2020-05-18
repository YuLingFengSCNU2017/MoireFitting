import numpy as np
import cv2

pixel_size = 256
random_point = 2


def makepoint(way=2):
    if way == 1:
        """way 1"""
        radi = 60
        pixel_center = int(pixel_size / 2)
        high1 = pixel_center + np.random.randint(-radi, radi, size=random_point)
        high2 = pixel_center + np.random.randint(-radi, radi, size=random_point)
    elif way == 2:
        """way 2"""
        pointmin, pointmax = 100, 156
        high1 = np.random.randint(pointmin, pointmax, size=random_point)
        high2 = np.random.randint(pointmin, pointmax, size=random_point)
    else:
        high1 = np.linspace(0, 255, 1, dtype='int')
        high2 = high1
    print(high1, '\n', high2)
    return high1, high2


def logplt(im0, getlog=1, windowname='unname'):
    im0 = np.abs(im0)
    if getlog == 1:
        im0plt = np.log(1 + im0) / np.log(2)
    else:
        im0plt = im0
    cv2.imshow('{}'.format(windowname), im0plt)


def realpic(im0):
    im1 = np.fft.fft2(im0)  # np.fft.ifftshift(im0)
    im2 = np.fft.fftshift(im1)  # np.fft.ifft2(im1)
    im3 = np.abs(im2)
    im3 = im2 / np.max(im3)
    im4 = np.arctan(np.imag(im2) / np.real(im2))
    im4 = np.nan_to_num(im4)
    # im4 = (im4 + np.pi / 2) / np.pi
    im4 = abs(im4) * 2 / np.pi
    print(im4.max())
    return im3, im4


def white(im0, expand=0):
    high1, high2 = makepoint()
    if expand == 0:
        for i in range(len(high1)):
            im0[high1[i], high2[i]] = 1.0
    else:
        for i in range(len(high1)):
            im0[high1[i] - expand:high1[i] + expand, high2[i] - expand:high2[i] + expand] = 1.0
    return im0


def show_pic(img):
    logplt(img, 1, 'fourie')
    rimg = realpic(img)
    logplt(rimg[0], 0, 'real_abs')
    logplt(rimg[1], 0, 'real_phase')
    cv2.waitKey(0)


def main():
    img = np.zeros([pixel_size, pixel_size], dtype='float')
    img = white(img)
    show_pic(img)


if __name__ == '__main__':
    main()
