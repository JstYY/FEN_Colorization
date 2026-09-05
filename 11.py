import os
import cv2
import numpy as np
from skimage import color
from PIL import Image

#对文件夹内的图片去除黑白图像，然后储存在目标文件夹内
# # 定义源文件夹和目标文件夹
# src_dir = 'E:\Python_Project\detr-main\coco\\val2017'
# dst_dir = 'E:\Python_Project\detr-main\coco\\val2017_color'
#
# # 检查源文件夹是否存在
# if not os.path.exists(src_dir):
#     print('源文件夹不存在，请检查路径是否正确！')
#     exit(0)
#
# # 创建目标文件夹
# os.makedirs(dst_dir, exist_ok=True)
#
# for filename in os.listdir(src_dir):
#     if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):  # 针对这些格式的图像处理
#         img_path = os.path.join(src_dir, filename)
#         image = cv2.imread(img_path, cv2.IMREAD_COLOR)  # 读取图像
#
#         # 判断是否为黑白图像
#         if len(image.shape) == 2:  # 黑白图像
#             print(f'忽略黑白图像: {img_path}')
#             continue
#         if image.shape[2] == 1:
#             print(f'忽略黑白图像: {img_path}')
#             continue
#
#         new_path = os.path.join(dst_dir, filename)
#         cv2.imwrite(new_path, image)  # 保存彩色图像到目标文件夹
#
# print('去除黑白图像完成!')
#对文件夹的图像全部转化为黑白图像
# 定义源文件夹和目标文件夹
src_dir = 'E:\Python_Project\detr-main\coco\\val2017'
dst_dir = 'E:\Python_Project\detr-main\coco\\val2017_gray'

# 检查源文件夹是否存在
if not os.path.exists(src_dir):
    print('源文件夹不存在，请检查路径是否正确！')
    exit(0)

# 创建目标文件夹
os.makedirs(dst_dir, exist_ok=True)

for filename in os.listdir(src_dir):
    if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):  # 针对这些格式的图像处理
        img_path = os.path.join(src_dir, filename)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)  # 读取图像

        # 转化为灰度图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        new_path = os.path.join(dst_dir, filename)
        cv2.imwrite(new_path, gray)  # 保存灰度图像到目标文件夹


print('转化为黑白图像完成!')