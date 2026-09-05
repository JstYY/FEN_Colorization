#给我一个程序对文件夹内的图片变为灰度图像替换原文件：
import os
import cv2


def convert_to_gray(folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 检查文件是否为图片文件
        if not os.path.isfile(file_path) or not any(
                file_path.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
            continue

        # 读取图像文件
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        # 将图像转换为灰度图像
        if image is not None:
            # 保存灰度图像，覆盖原文件
            cv2.imwrite(file_path, image)
            print(f"Converted {filename} to grayscale.")
        else:
            print(f"Failed to read {filename}.")


# 指定要转换的文件夹路径
folder_path = "E:\Python_Project\Color2Embed-main\data\mydata\\20001"

# 调用函数进行转换
convert_to_gray(folder_path)