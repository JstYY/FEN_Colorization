import cv2
import os
import numpy as np
import argparse
from tqdm import tqdm
import torch
from torch import nn
from torchvision.transforms import Resize
from torchvision import transforms
from torchvision.transforms import ToPILImage, Resize, ToTensor
from GCoNet.dataset import get_loader
from GCoNet.models.GCoNet import GCoNet
from GCoNet.util import save_tensor_img
from GCoNet.config import Config



class ImageProcessor:
    def __init__(self):
        # 模型初始化
        self.config = Config()
        self.device = torch.device("cuda")
        self.model = GCoNet().to(self.device)
        self.model_path = 'E:/Python_Project/Color2Embed-main/GCoNet/GCoNet.pth'
        gconet_dict = torch.load(self.model_path)
        self.model.load_state_dict(gconet_dict)
        self.model.eval()

        #


        # 确保预测目录存在
        # os.makedirs(self.pred_dir, exist_ok=True)


    def process_images(self, img_tensor, ref_tensor):
        self.img_tensor = img_tensor
        self.ref_tensor = ref_tensor
        self.target_size = img_tensor.size(2), img_tensor.size(3)
        colored_images = []

        inputs = img_tensor.to(self.device)
        gts = ref_tensor.to(self.device)

        with torch.no_grad():
            scaled_preds = self.model(inputs)[-1]

        for scaled_pred in scaled_preds:
            if self.config.db_output_refiner or not self.config.refine and self.config.db_output_decoder:
                res = nn.functional.interpolate(scaled_pred.unsqueeze(0), size=self.target_size, mode='bilinear',
                                                align_corners=True)
            else:
                res = nn.functional.interpolate(scaled_pred.unsqueeze(0), size=self.target_size, mode='bilinear',
                                                align_corners=True).sigmoid()
            res = res.cpu().squeeze(0)
            colored_images.append(res)

        return torch.stack(colored_images)
