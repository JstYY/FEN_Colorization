import numpy as np
import data.thinplate as tps
import cv2
import random
import math

# Reference : https://github.com/cheind/py-thin-plate-spline

def tps_transform(img, dshape=None):

    while True:
        point1 = round(random.uniform(0.3, 0.7), 2)
        point2 = round(random.uniform(0.3, 0.7), 2)
        range_1 = round(random.uniform(-0.25, 0.25), 2)
        range_2 = round(random.uniform(-0.25, 0.25), 2)
        if math.isclose(point1 + range_1, point2 + range_2):
            continue
        else:
            break

    c_src = np.array([
        [0.0, 0.0],
        [1., 0],
        [1, 1],
        [0, 1],
        [point1, point1],
        [point2, point2],
    ])

    c_dst = np.array([
        [0., 0],
        [1., 0],
        [1, 1],
        [0, 1],
        [point1 + range_1, point1 + range_1],
        [point2 + range_2, point2 + range_2],
    ])

    dshape = dshape or img.shape
    theta = tps.tps_theta_from_points(c_src, c_dst, reduced=True)
    grid = tps.tps_grid(theta, c_dst, dshape)
    mapx, mapy = tps.tps_grid_to_remap(grid, img.shape)
    return cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC)
#这个函数是用来对图像进行TPS变换的。
# TPS变换是一种非线性变换，它可以通过一组控制点来对图像进行形变。
# 在这个函数中，我们首先随机生成了两个控制点，然后计算了这两个控制点之间的变换。
# 最后，我们使用 OpenCV 的 remap 函数对图像进行变换。

