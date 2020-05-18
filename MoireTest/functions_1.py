
'''=========================================================================================
Date   :  2020.03.20
Author :  玩具反斗星
WeChat :  SCNU20172321114
Note   :  1.本程序用于手写的笔迹清晰化,请将照片放在本程序的同一目录，重命名为1.jpg。
          2.请确保运行环境正确，python3.0以上，并安装Numpy与OpenCV库。
          3.本程序处理手机拍照后的照片,请先确保图像足够清晰。对比度越高(字迹与背景颜色对比越明显)越好。
          4.参数过小（小于0）或过大均会导致数据溢出。请优先保证照片质量。
          5.输出的两个图片，挑其中一个清晰的即可
          
Help   :  1.按下1、2或3分别进入参数1、2、3的设置
            (去噪等级，自适应区间范围，自适应偏移量)
          2.进入设置后按下+或-调整参数大小
            (控制台会有提示)
          3.随时可以按Esc退出。当图片大小合适后，按回车完成。
========================================================================================='''

'''设置头文件和参数'''
import numpy as np
import cv2
nl,rgs,ecs = 3,3,10
# 去噪等级noiselevel，自适应区间范围ranges，自适应偏移量excursion

'''设置自适应区间'''    
rgs = 2 * rgs + 1 # 保证rgs为奇数

'''======================================================================================'''

'''定义函数'''

'''1,转化为灰度图；2,去椒盐噪声。'''
def gg(filename): # get gray
    '''输入文件名，提取灰度数据并去除椒盐噪声。输出图片的灰度。第二参数必须为奇数'''
    im = cv2.imread(filename) # 读取图片文件
    im = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY) # 转化为灰度图
    im = cv2.medianBlur(im,nl) # 去椒盐噪声
    return im

'''3,直方图均衡化；4，图像二值化。'''
def ssn(im): # spiced salt noise
    '''输入图片，均衡化并进行局部阈值自适应的图像二值化。输出图片。'''
    ssnim1 = cv2.equalizeHist(im)
    # im = cv2.threshold(im,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # 非区域自适应
    ssnim1 = cv2.adaptiveThreshold(
        ssnim1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,rgs,ecs)
    ssnim2 = cv2.adaptiveThreshold(
        im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,rgs,ecs)
    # 说明：第2、3个参数：阈值（此处用Otsu确定）和像素的目标灰度，
    # Binary：二值化，Otsu：最大类间方差法
    return ssnim1,ssnim2

'''======================================================================================'''

'''主程序'''
def main():
    '''提取图片源'''
    im1 = gg('1.jpg')

    '''按步骤执行，并输出图片'''
    im2 = ssn(im1)
    cv2.imwrite('2-01.jpg', im2[0])
    cv2.imwrite('2-02.jpg', im2[1])

    '''调整参数'''
    cv2.imshow('image', im2[1])
    OKset, numpara = 0, 1
    params = [nl, rgs, ecs]
    param_weight = [1, 2, 3]  # 调节的权重
    print('originparams=[{},{},{}]'.format(nl, rgs, ecs))
    while (OKset == 0):
        k = cv2.waitKey(0)
        if k == 27:  # wait for ESC key to exit
            cv2.destroyAllWindows()
            OKset = 1
            break
        elif k == 13:  # Set parameters: OK
            cv2.imwrite('2-01.jpg', im2[0])
            cv2.imwrite('2-02.jpg', im2[1])
            OKset = 1
            cv2.destroyAllWindows()
            print('Last parameters:', params)
            break
        elif k in [49, 50, 51]:  # ord('1','2','3')
            numpara = k - 49
            paramtext = ['noiselevel', '  ranges  ', ' excursion']
            print('=====set ', paramtext[numpara], '=====')
            continue
        elif k in [43, 45]:  # ord('+','-')
            dvalue = 44 - k  # -1 or 1
            params[numpara] = params[numpara] + dvalue * param_weight[numpara]
            nl, rgs, ecs = params  # 参数重新赋值
            im2 = ssn(im1)
            cv2.imshow('image', im2[1])
            cv2.imwrite('2-01.jpg', im2[0])
            cv2.imwrite('2-02.jpg', im2[1])
            print('parameters:', params)
            cv2.imwrite('2-03.jpg', np.bitwise_xor(im2[0], im2[1]))
        else:
            continue

if __name__ == "main":
    main()