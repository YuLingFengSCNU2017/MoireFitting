'''=========================================================================================
Date   ： 2020.02.24
Update ： 2020.03.20
Author ： 华南师范大学17届大创队伍（莫尔效应）
Note   ： 1、本程序用于莫尔条纹的精确检测与计数。
          2、请确保运行环境正确，在运行前请先测试图像是否清晰。
          3、本程序针对的对象为与水平成-45°≤α≤45°的莫尔条纹的竖直运动。
          4、莫尔条纹与水平面成倾斜角越接近45°越好（防止数据溢出），但是不能超过45°。
          5、坐标转换说明： (x,y) = (j,-i) 。条纹位于(x,y)系的第四象限内。
========================================================================================='''

'''设置头文件和参数'''
import numpy as np
from numpy import array
import cv2
import dealjpg as dj

'''主程序'''

'''提取图片源'''
im1 = dj.gg('1.jpg')
im2 = dj.gg('2.jpg')
'''按步骤执行，并输出中间图片'''
##im3 = dj.ssn(im1)
##cv2.imwrite('testjpg\\Python-ssn3.jpg',im3[1]) #输出图片的方式
##im3 = dj.ts(im3,10)
##cv2.imwrite('testjpg\\Python-ts3.jpg',im3) #输出图片的方式
##im4 = dj.ssn(im2)
##cv2.imwrite('testjpg\\Python-ssn4.jpg',im4[1]) #输出图片的方式
##im4 = dj.ts(im4,10)
##cv2.imwrite('testjpg\\Python-ts4.jpg',im4) #输出图片的方式
##data3 = dj.cf(im3)
##data4 = dj.cf(im4)
##data0 = dj.dtit(data3,data4)


im3 = dj.ts(im1, 10)
cv2.imwrite('testjpg\\Python-ts3.jpg', im3)  # 输出图片的方式
im4 = dj.ts(im2, 10)
cv2.imwrite('testjpg\\Python-ts4.jpg', im4)  # 输出图片的方式
data3 = dj.cf(im3)
data4 = dj.cf(im4)
data0 = dj.dtit(data3, data4)

'''输出结果'''
print('★== 位移：{}  像素\n★== 条纹间距：{}  像素\n★== 斜率：{}\n'.format(data0[0], data0[1], data0[2]))
if data0[1] != 0:
    print('★== 换算为莫尔条纹：移动了{}个莫尔条纹'.format(data0[0] / data0[1]))
