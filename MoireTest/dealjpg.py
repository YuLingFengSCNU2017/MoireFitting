"""=========================================================================================
Date   :  2020.03.20
Author :  玩具反斗星
WeChat :  SCNU20172321114
Note   ： 1、本程序用于莫尔条纹的精确检测与计数。
          2、请确保运行环境正确，在运行前请先测试图像是否清晰。
          3、本程序针对的对象为与水平成-45°≤α≤45°的莫尔条纹的竖直运动。
          4、莫尔条纹与水平面成倾斜角越接近45°越好（防止数据溢出），但是不能超过45°。
          5、坐标转换说明： (x,y) = (j,-i) 。条纹位于(x,y)系的第四象限内。
          
How To Set Parameters:
          1.按下1、2、3或4分别进入参数1、2、3、4的设置
            (去噪等级，自适应区间范围，自适应偏移量)
          2.进入设置后按下+或-调整参数大小
            (控制台会有提示)
          3.随时可以按Esc退出。当图片大小合适后，按回车完成。
========================================================================================="""

'''设置头文件和参数'''
import numpy as np
from numpy import array
import cv2

jpgfile = '1.jpg'
nl, rgs, ecs, kewd = 4, 37, 13, 0
# 去噪等级noise level,自适应区间范围ranges,自适应偏移量excursion,开运算核宽kernel width
rsp = 3  # 拟合预留像素reserved pixel
'''======================================================================================'''

'''定义函数'''
'''1,转化为灰度图；2,去椒盐噪声。'''


def gg(filename):  # get gray(salt and pepper noise)
    """
    输入文件名，提取灰度数据并去除椒盐噪声。输出图片的灰度。第二参数必须为奇数
    """
    im = cv2.imread(filename)  # 读取图片文件
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)  # 转化为灰度图
    im = cv2.medianBlur(im, 2 * nl + 1)  # 去椒盐噪声
    return im


'''3,直方图均衡化；4，图像二值化。'''


def ib(im, Get_equalizeHist=0):  # image binaryzation
    """
    输入图片1，均衡化并进行局部阈值自适应的图像二值化,输出图片1。
    或者再次处理图片1，进行局部阈值自适应的图像二值化,输出图片1和2。
    """
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


'''5,图像细化。'''  # Zhang-Suen算法


def ts(im, pixel=2, kewd2=kewd):  # thining step
    """
    输入一张灰度图，开始提取莫尔条纹的骨架。输出骨架的中间部分，裁剪一定像素。
    """
    kernel2 = np.ones((2 * kewd2 + 1, 2 * kewd2 + 1), np.uint8)  # 创建膨胀核
    im = cv2.dilate(im, kernel2)  # 膨胀
    delstep = [1, 1]  # 该变量用于指示循环是否继续。当[0,0]时，结束循环。
    step = 1  # 该变量用于指示当前标记点为步骤1还是步骤2
    # im[0,:]=im[im.shape[0]-1,:]=im[:,0]=im[:,im.shape[1]-1]=255 # cdt1若取到1，可加边框骨架。
    while (delstep != [0, 0]):  # 只有当delstep1&2都结束才停止细化
        #   for ii9 in range(200):
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
                '''判断四个条件:condition1~4'''
                [[p9, p2, p3], [p8, p1, p4], [p7, p6, p5]] = im0[i - 1:i + 2, j - 1:j + 2]
                pdata = [p2, p3, p4, p5, p6, p7, p8, p9]  # 将外围数据储存在pdata数组中
                cdt1 = 1 if 1 <= (sum(pdata)) <= 6 else 0
                cdt2 = 1 if (sum((1 - pdata[x - 1]) * pdata[x] for x in range(8))) == 1 else 0
                cdt3 = 1 - (p2 * p4 * pdata[2 + 2 * step])
                cdt4 = 1 - (p6 * p8 * pdata[4 - 2 * step])
                '''是否删除？'''
                cdt0 = cdt1 * cdt2 * cdt3 * cdt4  # cdt0=1的点符合条件，需要标记
                if cdt0 == 1:
                    countdelpoint += 1
                    im[i][j] = 0  # 在im标记板上确定是否删除
                    delstep = [1, 1]  # 触发继续循环指示
        cv2.imshow('pic0', im0)
        cv2.waitKey(0)
    im0 = im[pixel - 1:im.shape[0] - 2 * pixel - 1, pixel - 1:im.shape[1] - 2 * pixel - 1]  # 边框截取
    return im0


'''6,图像选点；7,最小二乘法拟合。'''


def cf(im, totalfit=3):  # catch point and fit line
    """
    输入骨架图与条纹数(默认3，最大5)，根据其中的像素进行直线拟合，
    并求出条纹间距。注意斜率必须存在。输出[斜率,[每条线的截距]]。
    """
    linedata = []  # 统计全部直线的点，元素是per_line
    perline = []  # 统计一条线的点，元素是点的坐标[x,y]=[j,-i]其中i向下，j向右
    point1 = []  # 记录第一个点的y值
    for p1i in range(im.shape[0] - 1):
        if im[p1i, 0] != 0: point1.append(p1i)
    ##    di = [0,1,-1] # 点必须连续
    i, j, k = 0, 0, 0  # 定义索引变量
    midline = int(len(point1) / 2)  # 取中间的那条线的序号
    lines = [point1[midline], point1[midline - 1], point1[midline + 1],
             point1[midline - 1], point1[midline + 1]]
    lines = lines[0:totalfit - 1]  # 中间的若干条线
    for i0 in lines:  # 若干条线逐一提取
        j, i = 0, i0  # 第一个点坐标已经给出
        perline = []  # 清空数组的值
        while (0 <= i < im.shape[0] - rsp) and (0 <= j < im.shape[1] - rsp):  # 点的坐标在范围内的时候
            perline.append([j, -i])  # 填入坐标(x,y)
            if im[i - 1, j + 1] == 1:
                i, j = i - 1, j + 1
            elif im[i + 1, j + 1] == 1:
                i, j = i + 1, j + 1
            elif im[i, j + 1] == 1:
                j = j + 1
            elif im[i - 1, j] == 1:
                j = j + 1
                while (im[i, j] == 0):
                    if (0 <= i - 1 < im.shape[0] - rsp):
                        i = i - 1  # 点的坐标在范围内的时候
                    else:
                        print('err - out range')
                        exit(0)
            elif im[i + 1, j] == 1:
                j = j + 1
                while (im[i, j] == 0):
                    if (0 <= i + 1 < im.shape[0] - rsp):
                        i = i + 1  # 点的坐标在范围内的时候
                    else:
                        print('err - out range')
                        exit(0)
            else:
                print('all zero, return error!')
                exit(0)
        linedata.append(perline)  # linedata里面有若干条线的数据
    '''从数据矩阵linedata中利用最小二乘法多项式拟合(polyfit)分别拟合出三条直线的斜率和截距'''
    linesKB_list = []
    for oneline in linedata:  # 对每一条线的数据都进行一个拟合
        oneline_array = array(oneline)
        linesKB_list.append(np.polyfit(oneline_array[:, 0], oneline_array[:, 1], 1))
    linesKB = array(linesKB_list)  # 转化为numpy好提取截距斜率

    slope = sum(list(linesKB[:, 0])) / len(list(linesKB[:, 0]))  # 平均斜率
    intercept = linesKB[:, 1]  # 各线截距
    return [slope, list(intercept)]


'''8,平均斜率与条纹间距；9,两个图之间的最小位移。'''


def dtit(pic1, pic2):  # distance & interval
    """
    输入的两个参数均为[斜率,[每条线的截距]](要求线数一致)，输出的是直线最小距离均值minD、
    莫尔条纹的间距MoireD、以及平均斜率aveK组成的数组[minD，MoireD，aveK]
    """
    aveK = 0.5 * pic1[0] + 0.5 * pic2[0]
    intercepts = []  # 储存莫尔条纹移动距离
    MoireDiss = []  # 储存莫尔条纹间距
    for intercept1 in pic1[1]:
        for intercept2 in pic2[1]:
            intercepts.append(abs(intercept1 - intercept2))  # 计算每一个线的距离
    for Moirenum in range(len(pic1[1]) - 1):
        MoireDiss.append(abs(pic1[1][Moirenum + 1] - pic1[1][Moirenum]))
        MoireDiss.append(abs(pic2[1][Moirenum + 1] - pic2[1][Moirenum]))
    Distances = array((sorted(intercepts))[0:(len(pic1[1]) - 1)])  # 取最小的若干条线计算位移
    slc = 1 / np.sqrt(1 + aveK ** 2)  # 定义倾斜系数slc为1/√1+k²
    minD = np.average(Distances) * slc  # 直线间距离 = 截距差*slc
    MoireD = np.average(MoireDiss) * slc  # 莫尔条纹间距 = 同一图相邻条纹截距 * slc
    return [minD, MoireD, aveK]


'''调整参数'''


def set_parameters(filename, param_weight=[1, 1, 2, 1]):
    """
    输入图像，调节的权重，开始调参。直接改变参数值，因此无返回值。
    """
    global nl, rgs, ecs, kewd
    img = gg(filename)  # 灰度化，去噪
    img = ib(img)  # 均衡化，二值化
    params = [nl, rgs, ecs, kewd]
    print('originparams=[{},{},{},{}]'.format(nl, rgs, ecs, kewd))
    OKset = 0
    while (OKset == 0):
        cv2.imshow('image', img)
        k = cv2.waitKey(0)
        if k == 27:  # wait for ESC key to exit
            cv2.destroyAllWindows()
            OKset = 1
            break
        elif k == 13:  # Set parameters: OK
            cv2.imwrite('{}.jpg'.format(params), img)
            OKset = 1
            cv2.destroyAllWindows()
            print('Last parameters:', params)
            break
        elif k in [49, 50, 51, 52]:  # ord('1','2','3','4')
            numpara = k - 49
            paramtext = [' NoiseLevel', '   Ranges  ', ' Excursion ', 'KernelWidth']
            print('====set===', paramtext[numpara], '====')
            continue
        elif k in [43, 45]:  # ord('+','-')
            dvalue = 44 - k  # -1 or 1
            params[numpara] = params[numpara] + dvalue * param_weight[numpara]
            nl, rgs, ecs, kewd = params  # 参数重新赋值
            img = gg(filename)  # 灰度化，去噪
            img = ib(img)  # 均衡化，二值化
            print('parameters:', params)
        else:
            continue


'''======================================================================================'''

'主函数'


def main():
    """
    提取图片源
    按步骤执行，并输出图片
    """
    set_parameters(jpgfile)
    im1 = gg(jpgfile)
    im2 = ib(im1)
    cv2.imwrite('2-01.jpg', im2)
    im2 = ts(im2, 0)
    cv2.imwrite('2-02.jpg', im2)
    # cv2.imwrite('2-1.jpg',im2[1])
    # set_parameters(im1,[1,1,1])


'主程序'
if __name__ == '__main__':
    main()
