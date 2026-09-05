import cv2
import os
import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms import Resize

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.hub.api import HubApi


def reshape_tensor(self, img_tensor, target_size):
    transform = Resize(target_size)  # target_size例如 (256, 256)
    return transform(img_tensor)  # 应用转换并返回结果
# YOUR_ACCESS_TOKEN = 'd695c830-867f-4603-85e8-6599e6106d12'
#
# api = HubApi()
# api.login(YOUR_ACCESS_TOKEN)
# # 用于取消设置的环境变量，这里您可能需要根据实际情况来设定
# os.environ['VLLM_USE_MODELSCOPE'] = '0'

class ImageColorizer:
    def __init__(self, output_folder1=None, output_folder2=None):
        self.img_colorization = pipeline(Tasks.image_colorization, model='damo/cv_ddcolor_image-colorization')  # 创建图像着色管道
        self.output_folder1 = output_folder1
        self.output_folder2 = output_folder2
        self.to_tensor = transforms.ToTensor()
        self.image_count = 0

        if output_folder1 is not None:
            os.makedirs(output_folder1, exist_ok=True)


    # 在类中定义reshape函数
    def reshape_tensor(self, img_tensor, target_size):
        transform = Resize(target_size)  # target_size例如 (256, 256)
        return transform(img_tensor)  # 应用转换并返回结果

    def colorize(self, img_batch_tensor, save_to_folder=False):#怎么将上述的output_folder传入到这个函数中?
        colored_images = []
        self.image_count = 0
        # 这个函数现在接受一个四维的批次张量
        if save_to_folder and self.output_folder1 is not None and img_batch_tensor.dim() == 4:
            for i, img_tensor in enumerate(img_batch_tensor):  #
                # 确保批次中的每个图像都是三维的
                if img_tensor.dim() != 3:
                    raise ValueError(
                        f'Each img_tensor in batch should be 3 dimensions (C, H, W), got {img_tensor.dim()}')

                img_np = img_tensor.cpu().numpy()

                # 先进行归一化, 然后转换为[H, W, C]格式
                if img_np.max() <= 1:
                    img_np = (img_np * 255).astype('uint8')
                img_np = np.transpose(img_np, (1, 2, 0))
                # cv2.imwrite('E:\Python_Project\Color2Embed-main\\test_datasets\gray2color\otest1\img_np.jpg', img_np)
                #将张量转化为numpy数组

                os.makedirs(self.output_folder2, exist_ok=True)
                image_count = self.image_count
                self.image_count += 1
                file_name = f'{i}.jpg'  # 为每张图像生成独特的文件名
                output_file = os.path.join(self.output_folder1, file_name)#将图片保存到指定的文件夹
                cv2.imwrite(output_file, img_np)#将图片保存到指定的文件夹

            # 定义数据文件夹路径和输出文件夹路径
            self.data_folder = self.output_folder1
                # 创建输出文件夹

            # 遍历数据文件夹中的图像文件
            for file_name in os.listdir(self.data_folder):
                file_path = os.path.join(self.data_folder, file_name)

                # 加载图像
                img = cv2.imread(file_path)

                # 使用图像着色管道进行图像上色
                result = self.img_colorization(img)

                # 生成输出文件路径
                output_file = os.path.join(self.output_folder2, file_name)

                # 保存结果图像
                cv2.imwrite(output_file, result[OutputKeys.OUTPUT_IMG])
                #这个循环内将图片转化为张量后拼接到一起
            target_size = (256, 256)
            for file_name in os.listdir(self.output_folder2):
                file_path = os.path.join(self.output_folder2, file_name)#将图片转化为张量
                img_np = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)#将图片转化为张量
                result_tensor = self.to_tensor(img_np)#将图片转化为张量
                result_tensor = self.reshape_tensor(result_tensor, target_size)  # 调整至目标大小
                colored_images.append(result_tensor)
                # Stack the list of tensors to form a batch and return
            return torch.stack(colored_images)
        #这里的张量是一个四维的张量，每一个维度分别是batch_size,channel,height,width



        else:
            raise ValueError("Unsupported tensor dimensions; expected a 4D tensor.")


# # 以下是示例用法
# img_path = "E:\Python_Project\Color2Embed-main/test_datasets\gray2color\own_test_data_gray/000_in.jpg"
# img_np = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
# img_tensor = torch.from_numpy(img_np).unsqueeze(0).permute(0, 3, 1, 2).float()
# colorizer = ImageColorizer(output_folder1="E:\Python_Project\Color2Embed-main\\test_datasets\gray2color\otest1",output_folder2="E:\Python_Project\Color2Embed-main\\test_datasets\gray2color\otest2")
# colored_image_tensor = colorizer.colorize(img_tensor, save_to_folder=True)
# colorizer = ImageColorizer(output_folder2="E:\Python_Project\Color2Embed-main\\test_datasets\gray2color\otest2")

# colored_image_tensor = colorizer.colorize(colored_image_tensor, save_to_folder=True)

#
# # 假设以下是您加载图像的 Tensor
# current_img_tensor = ...  # 当前图像 Tensor
# reference_img_tensor = ...  # 参考图像 Tensor
#
# # 初始化 STEGO_seg 模型
# stego_seg = STEGO_seg(cra=您的设置, n_clusters=27或您的设置)
#
# # 应用模型以获取当前图像和参考图像的语义分割信息
# cluster_value_cur, cluster_preds_cur = stego_seg.my_app(current_img_tensor)
# cluster_value_ref, cluster_preds_ref = stego_seg.my_app(reference_img_tensor)
#
# # 分离语义分割一致与不一致的区域
# consistent_mask, inconsistent_mask = separate_semantic_regions(cluster_preds_cur, cluster_preds_ref)
#
# # 一致和不一致的区域可根据 consistent_mask 和 inconsistent_mask 进一步处理
# # 示例代码`separate_semantic_regions` 如之前所示，或您可以根据需要进行自定义
#
# # 下一步，您将这些信息用于上色过程中的注意力机制或其他处理