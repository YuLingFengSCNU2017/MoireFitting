"""=========================================================================================
Date   :  2020.03.20
Author :  玩具反斗星
WeChat :  SCNU20172321114
Note   ： 1、本程序用于莫尔条纹的精确检测与计数。
          2、请确保运行环境正确，在运行前请先测试图像是否清晰。
          3、本程序针对的对象为与水平成-90°<α<90°的莫尔条纹的竖直运动。
          4、莫尔条纹与水平面成倾斜角越接近45°越好（防止数据溢出）。
          5、坐标转换说明： (x,y) = (j,-i) 。条纹位于(x,y)系的第四象限内。

How To Set Parameters:
          1.按下1、2、3、4或5分别进入参数1、2、3、4、5的设置
            (1、5去噪等级>0，2自适应区间范围>0，|3自适应偏移量|<255,4开运算核宽>0)
          2.进入设置后按下+或-调整参数大小
            (控制台会有提示)
          3.随时可以按Esc退出。当图片大小合适后，按回车完成。
========================================================================================="""

import numpy as np
from numpy import array
import cv2

jpg_file = 'testjpg\\1.jpg'
ts_del_pix = 25  # 删除的像素
nl, rgs, ecs, kewd, gl = 9, 42, 17, 7, 22
# 去噪等级noise level,自适应区间范围ranges,自适应偏移量excursion,
# 开运算核宽kernel width, 高斯去噪等级gauss level
rsp = 3  # 拟合预留像素reserved pixel
p_data = [nl, rgs, ecs, kewd, gl]
"""======================================================================================"""


def gg(filename):  # get gray(salt and pepper noise)
    """1,转化为灰度图；2,去椒盐噪声。
    输入文件名，提取灰度数据并去除椒盐噪声。输出图片的灰度。第二参数必须为奇数"""
    im = cv2.imread(filename)  # 读取图片文件
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)  # 转化为灰度图
    im = cv2.medianBlur(im, 2 * nl + 1)  # 去椒盐噪声
    im = cv2.GaussianBlur(im, (2 * gl + 1, 2 * gl + 1), 0)  # 去高斯噪音
    return im


def ib(im, Get_equalizeHist=0):  # image binaryzation
    """3,直方图均衡化；4，图像二值化。
    输入图片1，均衡化并进行局部阈值自适应的图像二值化,输出图片2。
    或者再次处理图片1，进行局部阈值自适应的图像二值化,输出图片2和3。"""
    ibim1 = cv2.equalizeHist(im)
    # im = cv2.threshold(im,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # 非区域自适应
    ibim1 = cv2.adaptiveThreshold(
        ibim1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, rgs * 2 + 1, ecs)
    # 说明：第2、3个参数：阈值（此处用Otsu确定）和像素的目标灰度，rgs*2+1是保证其为奇数
    # Binary：二值化，Otsu：最大类间方差法
    kernel = np.ones((2 * kewd + 1, 2 * kewd + 1), np.uint8)  # 创建开运算的核
    ibim1 = cv2.morphologyEx(ibim1, cv2.MORPH_OPEN, kernel)  # 开运算
    if Get_equalizeHist == 0:
        return ibim1
    else:
        ibim2 = cv2.adaptiveThreshold(
            im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, rgs * 2 + 1, ecs)
        ibim2 = cv2.morphologyEx(ibim2, cv2.MORPH_OPEN, kernel)  # 开运算
        return ibim1, ibim2


def ts(im, pixel=2):  # Zhang-Suen算法 thining step
    """5,图像细化。
    输入一张灰度图，开始提取莫尔条纹的骨架。输出骨架的中间部分，裁剪一定像素。"""
    im = im[pixel:im.shape[0] - 2 * pixel, pixel:im.shape[1] - 2 * pixel]  # 边框截取
    delstep = [1, 1]  # 该变量用于指示循环是否继续。当[0,0]时，结束循环。
    step = 2  # 该变量用于指示当前标记点为步骤1还是步骤2
    # im[0,:]=im[im.shape[0]-1,:]=im[:,0]=im[:,im.shape[1]-1]=255 # cdt1若取到1，可加边框骨架。
    while delstep != [0, 0]:  # 只有当delstep1&2都结束才停止细化
        step = 3 - step  # 1:step1, 2:step2
        delstep[step - 1] = 0  # 继续循环指示归0
        im0 = im / 255  # 在im0判据板上判断，在im标记板上确定是否删除
        countdelpoint = 0  # 计数
        countpixel = 0
        jump = 0
        for i in range(im0.shape[0]):
            if (i * (i - im0.shape[0] + 1)) == 0:
                continue
            for j in range(im0.shape[1]):
                countpixel += 1
                if (j * (j - im0.shape[1] + 1)) * (im0[i][j]) == 0:
                    jump += 1
                    continue
                """判断四个条件:condition1~4"""
                [[p9, p2, p3], [p8, p1, p4], [p7, p6, p5]] = im0[i - 1:i + 2, j - 1:j + 2]
                pdata = [p2, p3, p4, p5, p6, p7, p8, p9]  # 将外围数据储存在pdata数组中
                cdt1 = 1 if 1 <= (sum(pdata)) <= 6 else 0  # 2<=s(p)<=6
                cdt2 = 1 if (sum((1 - pdata[x - 1]) * pdata[x] for x in range(8))) == 1 else 0
                cdt3 = 1 - (p2 * p4 * pdata[2 + 2 * step])
                cdt4 = 1 - (p6 * p8 * pdata[4 - 2 * step])
                """是否删除？"""
                cdt0 = cdt1 * cdt2 * cdt3 * cdt4  # cdt0=1的点符合条件，需要标记
                if (cdt0 == 1):
                    countdelpoint += 1
                    im[i][j] = 0  # 在im标记板上确定是否删除
                    delstep = [1, 1]  # 触发继续循环指示

    im0 = im[pixel:im.shape[0] - 2 * pixel, pixel:im.shape[1] - 2 * pixel]  # 边框截取
    cv2.imshow('pic0', im0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return im0


def cf(im, direction, totalfit=4):  # catch point and fit line
    """6,图像选点；7,最小二乘法拟合。  totalfit必须是偶数
    输入骨架图、线走向与条纹数(默认3)，根据其中的像素进行直线拟合，
    上行、平行、下行（斜率正、接近0、负）分别对应：direction=1，0，-1
    并求出条纹间距。注意斜率必须存在。输出[斜率,[每条线的截距]]。"""
    im = im / im.max()  # 归一化
    linedata = []  # 统计全部直线的点，元素是per_line
    point1 = np.array([], dtype='int32')  # 记录第一个点的y值
    for p1i in range(im.shape[0] - 1):
        if im[p1i, 0] != 0 and sum(im[p1i - 1:p1i + 1, 0:1]) == 1:
            point1 = np.append(p1i, point1)
    midline = int(len(point1) / 2)  # 取中间的那条线的序号
    natary = np.arange(0, totalfit)  # natural array:0,1,2,3……
    deltaline = np.int32(0.5 * ((natary % 2) - natary) + natary * (natary % 2))
    lines = point1[midline - deltaline]
    print('lines=', lines, '\ndeltaline=', deltaline)
    for i00 in range(len(lines)):  # 若干条线逐一提取
        i0, j, i = lines[i00], 0, lines[i00]  # 第一个点坐标已经给出
        perline = []  # 清空数组的值。统计一条线的点，元素是点的坐标[x,y]=[j,-i]其中i向下，j向右
        while (0 <= i < im.shape[0] - rsp) and (0 <= j < im.shape[1] - rsp):  # 点的坐标在范围内的时候
            perline.append([j, -i])  # 填入坐标(x,y)
            if im[i - 1, j + 1] == 1 and direction >= 0:
                i, j = i - 1, j + 1
                continue
            elif im[i + 1, j + 1] == 1 and direction <= 0:
                i, j = i + 1, j + 1
                continue
            elif im[i, j + 1] == 1:
                j = j + 1
                continue
            elif (im[i - 1, j] == 1) and direction >= 0:
                j = j + 1
                while im[i, j] == 0:
                    if (0 <= i < im.shape[0] - rsp) and (0 <= j < im.shape[1] - rsp):
                        i = i - 1  # 点的坐标在范围内的时候
                    else:
                        print('err - out range: only ', len(perline))
                        break
            elif (im[i + 1, j] == 1) and (j != 0) and direction <= 0:
                j = j + 1
                while im[i, j] == 0:
                    if (0 <= i < im.shape[0] - rsp) and (0 <= j < im.shape[1] - rsp):
                        i = i + 1  # 点的坐标在范围内的时候
                    else:
                        print('out : only ', len(perline), 'if too small : error')
                        break
            else:
                print('Found an isolated point!')
                break
        linedata.append(perline)  # linedata里面有若干条线的数据
    # 从数据矩阵linedata中利用最小二乘法多项式拟合(polyfit)分别拟合出三条直线的斜率和截距
    linesKB_list = []
    for oneline in linedata:  # 对每一条线的数据都进行一个拟合
        oneline_array = array(oneline)
        linesKB_list.append(np.polyfit(oneline_array[:, 0], oneline_array[:, 1], 1))
    linesKB = array(linesKB_list)  # 转化为numpy好提取截距斜率

    slope = sum(list(linesKB[:, 0])) / len(list(linesKB[:, 0]))  # 平均斜率
    intercept = linesKB[:, 1]  # 各线截距
    return [slope, list(intercept)]


def dtit(dat1, dat2, totalfit):  # distance & interval
    """8,平均斜率与条纹间距；9,两个图之间的最小位移。
    输入的前两个参数均为[斜率,[每条线的截距]](要求线数为偶数,pic2-移动后)，
    输出的是直线最小距离均值minD、莫尔条纹的间距moire_diss、平均斜率aveK组成的数组[minD，MoireD，aveK]"""
    ave_k = 0.5 * dat1[0] + 0.5 * dat2[0]  # 斜率的平均值
    slc = 1 / np.sqrt(1 + ave_k ** 2)
    # 定义倾斜系数slc为1/√1+k²,直线间距离=截距差*slc,莫尔条纹间距=同一图相邻条纹截距*slc
    # 莫尔条纹间距
    moire_diss = 0  # 储存莫尔条纹间距
    dif = int(totalfit / 2)  # 逐差法求直线间距
    for dats in [dat1[1], dat2[1]]:
        dats.sort(reverse=1)
        moire_diss += 0.5 * slc * ((sum(dats[0:dif]) - sum(dats[dif:2 * dif])) / (dif ** 2))  # 两个间隔取平均值
    # 莫尔条纹移动的距离
    ave_diss = slc * (np.average(dat2[1]) - np.average(dat1[1]))  # 两组点的各自平均值求移动的距离
    if ave_diss >= 0.5 * moire_diss:
        min_diss = ave_diss - moire_diss
    elif ave_diss <= -0.5 * moire_diss:
        min_diss = ave_diss + moire_diss
    else:
        min_diss = ave_diss
    return [-min_diss, moire_diss, ave_k]  # 负号把i坐标空间变成y坐标空间，向上移动


def set_parameters(filename, param_data, param_weight=(1, 1, 2, 1, 1)):
    """调整参数
    输入图像，可自定义默认参数，调节的权重，开始调参。直接改变参数值，因此无返回值。"""
    global nl, rgs, ecs, kewd, gl
    nl, rgs, ecs, kewd, gl = param_data  # 参数传递
    img = gg(filename)  # 灰度化，去噪
    img = ib(img)  # 均衡化，二值化
    params = [nl, rgs, ecs, kewd, gl]
    print('originparams=[{},{},{},{},{}]'.format(nl, rgs, ecs, kewd, gl))
    numpara = 3  # 默认设置参数3，因为它是唯一可以为负的
    while 1:
        cv2.imshow('image', img)
        k = cv2.waitKey(0)
        if k == 27:  # wait for ESC key to exit
            cv2.destroyAllWindows()
            break
        elif k == 13:  # Set parameters: OK
            cv2.destroyAllWindows()
            print('Last parameters:', params)
            break
        elif k in [49, 50, 51, 52, 53]:  # ord('1','2','3','4','5')
            numpara = k - 49
            paramtext = [' NoiseLevel', '   Ranges  ', ' Excursion ', 'KernelWidth', 'GaussianLevel']
            print('=====\n   【   set  ', paramtext[numpara], '   】')
            continue
        elif k in [43, 45]:  # ord('+','-')
            dvalue = 44 - k  # -1 or 1
            params[numpara] += dvalue * param_weight[numpara]
            nl, rgs, ecs, kewd, gl = params  # 参数重新赋值
            img = gg(filename)  # 灰度化，去噪
            cv2.imshow('ib', img)
            cv2.imshow('gg', img * 255)
            img = ib(img)  # 均衡化，二值化
            print('parameters:', params)
        else:
            continue


"""======================================================================================"""

"""主函数"""


def main():
    """提取图片源"""
    """按步骤执行，并输出图片"""
    set_parameters(jpg_file,p_data)
    im1 = gg(jpg_file)
    im2 = ib(im1)
    cv2.imwrite('2-1.jpg', im2)
    im2 = ts(im2, ts_del_pix)
    cv2.imwrite('2-2.jpg', im2)


"""主程序"""
if __name__ == '__main__':
    main()
