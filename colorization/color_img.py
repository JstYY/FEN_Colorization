import cv2
import torch
from torchvision import models, transforms
from PIL import Image
import os
from glob import glob
import numpy as np
# 加载预训练的DeepLabV3模型
model = models.segmentation.deeplabv3_resnet101(
    weights=models.segmentation.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1).eval()
def apply_semantic_segmentation(img):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)
    return output_predictions
# 设置显示窗口大小
cv2.namedWindow('Masked IN Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Masked IN Image', 256, 256)
def resize_image(image, size=(512, 512)):
    # 图像重置大小
    return image.resize(size, Image.BILINEAR)

def process_image_pairs(folder_path, resize_to=None):
    # 读取文件夹内的所有'in'和'ref'图像
    in_images = sorted(glob(os.path.join(folder_path, '*_in.jpg')))
    ref_images = sorted(glob(os.path.join(folder_path, '*_ref.jpg')))

    # 确保图像是成对出现
    # assert len(in_images) == len(ref_images), "The number of 'in' and 'ref' images must be the same"

    for in_path, ref_path in zip(in_images, ref_images):
        # 读取 'in' 和 'ref' 图像进行语义分割
        input_image = Image.open(in_path).convert("RGB")
        reference_image = Image.open(ref_path).convert("RGB")

        # 如果需要，调整图像的大小
        if resize_to is not None:
            input_image = resize_image(input_image, size=resize_to)
            reference_image = resize_image(reference_image, size=resize_to)

        in_segmentation = apply_semantic_segmentation(input_image)
        ref_segmentation = apply_semantic_segmentation(reference_image)

        # 创建相同语义信息的掩膜
        mask = (in_segmentation == ref_segmentation).numpy().astype(np.uint8)

        # 应用掩膜并绘制结果，可以适当增加逻辑以保存或展示掩膜
        original_in_image = cv2.imread(in_path)
        resized_in_image = cv2.resize(original_in_image, resize_to, interpolation=cv2.INTER_LINEAR)
        mask_color = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
        result_image = cv2.addWeighted(resized_in_image, 0.5, mask_color, 0.5, 0)

        # 展示结果
        cv2.imshow('Masked IN Image', result_image)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
if __name__ == '__main__':
    folder_path = 'E:\Python_Project\Color2Embed-main\data\\000-599'
    resize_dims = (256, 256)  # 设置为你需要的尺寸
    process_image_pairs(folder_path, resize_to=resize_dims)