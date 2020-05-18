import numpy as np
import cv2


def flo(img):
    fimg = np.fft.fft2(img)
    fsimg = np.fft.fftshift(fimg)
    afsimg = np.abs(fsimg) / np.max(np.abs(fsimg))
    return fsimg, afsimg


def getpic(x_coefficient, y_coefficient, linenum):
    pic = np.ones([256, 256])
    for x in range(pic.shape[0]):
        for y in range(pic.shape[1]):
            pic[x, y] = 0.5 + 0.5 * np.cos((x_coefficient * x + y_coefficient * y) * np.pi * linenum / 512)
            pic = pic / np.max(pic)
            cv2.imshow('x:{},y:{}'.format(x_coefficient, y_coefficient), pic)
    return pic


def main():
    pic1 = getpic(3, 4, 16)
    pic2 = getpic(4, 3, 16)
    pic3 = np.array((pic1, pic2)).min(axis=0)
    # pic3 = np.minimum(pic1, pic2)
    cv2.imshow('real space', pic3)
    pic3f = flo(pic3)
    cv2.imshow('fourier space', pic3f[1])
    picmoire = getpic(1, -1, 16)
    picmoiref = flo(picmoire)
    cv2.imshow('moirefour space', picmoiref[1])
    print('expect:', np.nonzero(pic3f[1]))
    print('moire3:', np.nonzero(picmoiref[1]))

    cv2.waitKey(0)


if __name__ == '__main__':
    main()
