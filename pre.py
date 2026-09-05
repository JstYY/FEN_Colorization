#把文件夹内的图片reshape成256*256大小,包括所有子文件夹

import os
import cv2
import numpy as np
from PIL import Image

# 定义源文件夹和目标文件夹
src_dir = 'E:\Python_Project\Color2Embed-main\\results\gray2color\exemplar_based\Color2Embed_kk_5.5w'
dst_dir = 'E:\Python_Project\Color2Embed-main\\results\gray2color\exemplar_based\Color2Embed_kk_5.5w'

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
        if filename == '000_in.png':

            # 调整图像大小为256*256
            image = cv2.resize(image, (732, 552))
        elif filename == '001_in.png':
            # 调整图像大小为256*256
            image = cv2.resize(image, (733, 475))
        elif filename == '002_in.png':
            image = cv2.resize(image, (732, 353))
        else:
            image= cv2.resize(image, (730, 422))
        new_path = os.path.join(dst_dir, filename)
        cv2.imwrite(new_path, image)  # 保存调整大小后的图像到目标文件夹

print('调整大小完成!')
