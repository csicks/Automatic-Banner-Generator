# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
# import numpy as np
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
#
# # Make data.
# X = np.arange(-5, 5, 0.25)
# Y = np.arange(-5, 5, 0.25)
# X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X ** 2 + Y ** 2)
# Z = np.sin(R)
#
# # Plot the surface.
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
#
# # Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#
# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)
#
# plt.show()

#-*- coding: UTF-8 -*-

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


def draw_test():

    #生成深蓝色绘图画布
    array = np.ndarray((480, 640, 3), np.uint8)

    array[:, :, 0] = 0
    array[:, :, 1] = 0
    array[:, :, 2] = 100

    image = Image.fromarray(array)

    #创建绘制对象
    draw = ImageDraw.Draw(image)

    #绘制直线
    draw.line((20, 20, 150, 150), 'cyan')

    #绘制矩形
    draw.rectangle((100, 200, 300, 400), 'black', 'red')

    #绘制弧
    draw.arc((100, 200, 300, 400), 0, 180, 'yellow')
    draw.arc((100, 200, 300, 400), -90, 0, 'green')

    #绘制弦
    draw.chord((350, 50, 500, 200), 0, 120, 'khaki', 'orange')

    #绘制圆饼图
    draw.pieslice((350, 50, 500, 200), -150, -30, 'pink', 'crimson')

    #绘制椭圆
    draw.ellipse((350, 300, 500, 400), 'yellowgreen', 'wheat')
    #外切矩形为正方形时椭圆即为圆
    draw.ellipse((550, 50, 600, 100), 'seagreen', 'skyblue')

    #绘制多边形
    draw.polygon((150, 180, 200, 180, 250, 120, 230, 90, 130, 100), 'olive', 'hotpink')

    #绘制文本
    font = ImageFont.truetype("consola.ttf", 40, encoding="unic")#设置字体
    draw.text((100, 50), u'Hello World', 'fuchsia', font)

    image.show()

    return

draw_test()