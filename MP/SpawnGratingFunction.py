"""
一个生成光栅和莫尔条纹仿真的函数
将会得到1.jpg和2.jpg
data.dat中的数据：
1~5   :  moire间距(像素),斜率1,斜率2,垂直运动距离(像素)1,距离2,
6~11  :  a,b,c,d,新c,新d,
12~15 :  光栅水平位移,线数参数,像素参数,光栅转动角度
"""
import numpy as np
import cv2
import skimage.util as skut

a, b, c, d, distance, = 0.3, 4.0, 0.938083, 3.9, 14.142
linenum, pix, xita = 60, 500, (np.pi * 0 / 180)  # 25 300
"""栅线方程：(a*x+b*y)*(pi/2)*(linenum/pix)=2*m*pi"""
total_data1 = [a, b, c, d]  # 光栅1的x，y系数对应a，b，光栅2的对应c，d
total_data2 = [distance, linenum, pix, xita]  # 第三个光栅相对第二个的水平位移，线数参数，像素参数，光栅运动后的转动角误差


def getpic(x_coefficient, y_coefficient, _distance=0.0, _linenum=16, _pix=300, _xita=0, get_new_ab=0):
    """x系数,y系数,线数，像素，偏移角，X位移。两个系数相对大小决定斜率。如果出错，请检查斜率是否存在"""
    kp = (np.tan(_xita) + (-x_coefficient / y_coefficient)) / (
            1 - np.tan(_xita) * (-x_coefficient / y_coefficient))  # 新的斜率
    t = np.sqrt(x_coefficient ** 2 + y_coefficient ** 2)  # 计算系数t
    y_coefficient = t / np.sqrt(1 + kp ** 2)
    x_coefficient = -t * kp / np.sqrt(1 + kp ** 2)  # 计算新的系数
    iv, jv = np.meshgrid(np.arange(0, _pix, 1), np.arange(0, _pix, 1))
    pic = 0.5 + 0.5 * np.cos((x_coefficient * (jv - _distance) - y_coefficient * iv) * np.pi * _linenum / (2 * _pix))
    pic = pic / np.max(pic)  # [i,j]=[-y,x],[x, y] = [j, -i]
    if get_new_ab == 0:
        return pic
    else:
        new_ab = [x_coefficient, y_coefficient]
        return pic, new_ab


def getpim(im1, im2):
    im12 = np.array((im1, im2))
    im12 = im12.min(axis=0)
    im12 = cv2.GaussianBlur(im12, (5, 5), 0)
    im12 = skut.random_noise(im12, 'gaussian', var=0.05, mean=0.8)  # 高斯噪声方差，平均值
    im12 = skut.random_noise(im12, 's&p', amount=0.05)  # 椒盐噪声比例
    return im12


def outdata_cos_raster(c2, d2):
    """返回莫尔条纹的间距，莫尔条纹的斜率，莫尔条纹移动的距离。
    光栅方程：(a*x+b*y)*(pi/2)*(linenum/pix)=2*m*pi"""
    moire_moire = 4 * pix / (linenum * np.sqrt((c - a) ** 2 + (b - d) ** 2))  # 莫尔条纹间距
    moire_slope1 = (d - b) / (a - c)
    moire_slope2 = (d2 - b) / (a - c2)  # 莫尔条纹的斜率
    moire_distance1 = distance * linenum * moire_moire * c / (4 * pix)  # 莫尔条纹移动距离
    moire_distance2 = distance * linenum * moire_moire * c2 / (4 * pix)
    return [moire_moire, moire_slope1, moire_slope2, moire_distance1, moire_distance2]


def main():
    pic1 = getpic(a, b, _distance=0.0, _linenum=linenum, _pix=pix)
    pic2 = getpic(c, d, _distance=0.0, _linenum=linenum, _pix=pix)
    pic12 = getpim(pic1, pic2)
    pic3, [new_c, new_d] = getpic(c, d, _distance=distance, _linenum=linenum, _pix=pix, _xita=xita, get_new_ab=1)
    pic13 = getpim(pic1, pic3)
    cv2.imwrite('testjpg\\pic12.jpg', pic12 * 255)
    cv2.imwrite('testjpg\\pic13.jpg', pic13 * 255)

    total_data0 = outdata_cos_raster(new_c, new_d)

    set_param = open('data.dat', 'w')
    total_data = total_data0 + total_data1 + [new_c, new_d] + total_data2
    for data_text in total_data:
        set_param.write(str(data_text) + '\n')
        # print(data_text)
    set_param.close()

    cv2.imshow('pic1.jpg', pic12)
    cv2.imshow('pic2.jpg', pic13)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
