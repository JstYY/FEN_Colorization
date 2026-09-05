

import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os
from modelscope.hub.api import HubApi

YOUR_ACCESS_TOKEN = 'd695c830-867f-4603-85e8-6599e6106d12'

api = HubApi()
api.login(YOUR_ACCESS_TOKEN)
#unset VLLM_USE_MODELSCOPE
# 创建图像着色管道
img_colorization = pipeline(Tasks.image_colorization, model='damo/cv_ddcolor_image-colorization')

# 定义数据文件夹路径和输出文件夹路径
data_folder = "E:\Python_Project\Color2Embed-main\\test_datasets\gray2color\\test1111"
output_folder = "E:\Python_Project\Color2Embed-main\\test_datasets\gray2color\\test111"

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 遍历数据文件夹中的图像文件
for file_name in os.listdir(data_folder):
    file_path = os.path.join(data_folder, file_name)

    # 加载图像
    img = cv2.imread(file_path)

    # 使用图像着色管道进行图像上色
    result = img_colorization(img)

    # 生成输出文件路径
    output_file = os.path.join(output_folder, file_name)

    # 保存结果图像
    cv2.imwrite(output_file, result[OutputKeys.OUTPUT_IMG])