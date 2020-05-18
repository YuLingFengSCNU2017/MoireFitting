"""=========================================================================================
Date   :  2020.03.20
Author :  玩具反斗星
WeChat :  SCNU20172321114
Note   :  1、本程序用于莫尔条纹的精确检测与计数。
          2、请确保运行环境正确，在运行前请先测试图像是否清晰。
          3、本程序针对的对象为与水平成-90°<α<90°的莫尔条纹的竖直运动。
          4、莫尔条纹与水平面成倾斜角越接近45°越好(防止数据溢出)。
          5、坐标转换说明： (x,y) = (j,-i) 。条纹位于(x,y)系的第四象限内。
★★★★★ 6、调节参数的时候，需要保证不能断线！！即使拍摄更大的线间距！
========================================================================================="""
# 设置头文件和参数
import cv2
import numpy as np
import MP.functions as fun

# 设置图片文件位置，加载图片的理论值
im1, im2 = 'testjpg\\pic12.jpg', 'testjpg\\pic13.jpg'
datatxt = open('data.dat', 'r')
datas = datatxt.readlines()
datatxt.close()
mm0, mk1, mk2, md1, md2 = [float(datas[i]) for i in range(5)]
mfai0_err = 0.5 * abs(np.arctan(mk1) - np.arctan(mk2))
md_ave, md_err = 0.5 * (md1 + md2), 0.5 * abs(md1 - md2)
mm_ave, mm_err = md_ave / mm0, md_err / mm0

is_even_num, total_fit = 1, 5
while is_even_num:
    try:
        total_fit = int(input('请输入要拟合的线数，必须为【正偶数】：'))
        is_even_num = total_fit % 2 if total_fit > 1 else 1
    except ValueError:
        is_even_num = 1


def getdata(picname, filename='unnamed'):
    param_data = [4, 41, -11, 2, 19]
    fun.set_parameters(picname, param_data)
    pic = fun.gg(picname)
    pic = fun.ib(pic)
    pic = fun.ts(pic, int(0.02 * float(datas[13])))
    cv2.imwrite('testjpg\\{}bones.jpg'.format(filename), pic)
    data_sets = fun.cf(pic, 1, total_fit)
    return data_sets


def main():
    """处理数据"""
    data1 = getdata(im1, 'Py-ts1')
    data2 = getdata(im2, 'Py-ts2')
    print('\ndata1:  ', data1, '\ndata2:  ', data2, '\n')
    data0 = fun.dtit(data1, data2, total_fit)

    '''输出结果'''
    print('\n【沿条纹法线方向上移(-0.5)~(0.5)个条纹】')

    print('★== Moire条纹上移：{:.2f}  像素;'.format(data0[0]))
    print('相对理论值1的误差:({:.2f}%)  相对理论值2的误差:({:.2f}%)'.format((data0[0] - md_ave) * 100 / data0[0],
                                                            (data0[0] - md_ave + mm0) * 100 / data0[0]))
    print('  ==   上移 理论值1：{:.2f}(±{:.3})'.format(md_ave, md_err))
    print('  ==   下移 理论值2：{:.2f}(±{:.3})'.format(md_ave - mm0, md_err))
    print('\n★== Moire条纹间距：{:.2f}  像素;误差({:.2f}%)'.format(data0[1], (data0[1] - mm0) * 100 / data0[1]))
    print('  ==        理论值：{:.2f} '.format(mm0))
    fai, fai0 = np.arctan(data0[2]), 0.5 * (np.arctan(mk1) - np.arctan(mk2))
    print('\n★== Moire条纹倾角：{:.3} ;误差({:.3}%)'.format(fai, (fai - fai0) * 100 / fai0))
    print('  ==        理论值：{:.3}(±{:.3})'.format(fai0, mfai0_err))

    if data0[1] != 0:
        print('\n★== 换算为莫尔条纹：移动了 【{:.3}】 个莫尔条纹，误差({:.3}%)'.format(data0[0] / data0[1],
                                                                  ((data0[0] / data0[1]) - mm_ave) * 100 / (
                                                                          data0[0] / data0[1])))
        print('  == 莫尔条纹移动理论值：  {:.3}(±{:.3})   个莫尔条纹  '.format(mm_ave, mm_err))
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
