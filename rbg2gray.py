#这个程序可以将一个文件夹内的彩色图像转变为灰度图像并且储存于另一个文件夹内，保持图片名称不变：
import os
from PIL import Image

# 设置原始图片所在的文件夹路径
input_folder_path = 'E:\Python_Project\DDColor-master\data_list\ILSVRC2012_img_val_gray'
# 设置存储灰度图片的目标文件夹路径
output_folder_path = 'E:\Python_Project\DDColor-master\data_list\ILSVRC2012_img_val_gray1'

# 确保输出文件夹存在
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# 获取原始文件夹中所有的文件名
image_filenames = [f for f in os.listdir(input_folder_path) if os.path.isfile(os.path.join(input_folder_path, f))]

# 遍历所有文件，转换为灰度图并保存
for image_filename in image_filenames:
    image_path = os.path.join(input_folder_path, image_filename)

    # 打开图片并转换为灰度
    image = Image.open(image_path).convert('L')

    # 保存转换后的灰度图片到目标文件夹中，保持文件名不变
    image.save(os.path.join(output_folder_path, image_filename))

print('所有图片都已转换为灰度图并存储到目标文件夹。')

