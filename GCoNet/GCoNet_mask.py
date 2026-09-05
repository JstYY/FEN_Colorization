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
from dataset import get_loader
from models.GCoNet import GCoNet
from util import save_tensor_img
from config import Config


def resize_tensor(img_tensor, target_size):
    # 假设img_tensor是一个4维张量: [N, C, H, W]
    # 将其每一个张量转换为PIL图像，调整大小后再转换回Tensor
    transformer = transforms.Compose([
        ToPILImage(),  # 将其转换为PIL图像
        Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR),  # 调整大小
        ToTensor(),  # 再次转换为Tensor
    ])

    resized_tensors = [transformer(img_tensor[i].cpu()).unsqueeze(0) for i in range(img_tensor.size(0))]
    return torch.cat(resized_tensors, dim=0)  # 再次拼接回4维张量



def test_dataset_folder(root_dir, pred_dir, size, device, model,target_size):
    config = Config()

    test_img_path = os.path.join(root_dir, 'images')
    test_gt_path = os.path.join(root_dir, 'references')
    saved_root = pred_dir

    test_loader = get_loader(
        test_img_path, test_gt_path, size, 1, istrain=False, shuffle=False, num_workers=8, pin=True)

    os.makedirs(saved_root, exist_ok=True)
    colored_images = []
    for batch in tqdm(test_loader):
        inputs = batch[0].to(device).squeeze(0)#将batch[0]转换为tensor并且去掉第一个维度
        gts = batch[1].to(device).squeeze(0)#将batch[1]转换为tensor并且去掉第一个维度
        subpaths = batch[2]#subpaths是一个列表
        ori_sizes = batch[3]#ori_sizes是一个列表
        with torch.no_grad():
            scaled_preds = model(inputs)[-1]#

        num = len(scaled_preds)
        for inum in range(num):
            subpath = os.path.splitext(subpaths[inum][0])[0] + '.jpg'  # output in jpg format
            ori_size = (ori_sizes[inum][0].item(), ori_sizes[inum][1].item())
            if config.db_output_refiner or (not config.refine and config.db_output_decoder):#如果config.db_output_refiner为真或者config.refine为假且config.db_output_decoder为真
                res = nn.functional.interpolate(scaled_preds[inum].unsqueeze(0), size=ori_size, mode='bilinear',
                                                align_corners=True)
            else:
                res = nn.functional.interpolate(scaled_preds[inum].unsqueeze(0), size=ori_size, mode='bilinear',
                                                align_corners=True).sigmoid()
            save_tensor_img(res, os.path.join(saved_root, subpath))#res是一个张量，将张量转化为图片并且保存
            #将所有的张量按照第一个维度拼接为一个张量
            colored_images.append(res)
    # 上面的for循环结束后，在这里执行resize（如果需要）并堆叠张量
    resized_tensors = [resize_tensor(img, target_size) for img in colored_images]
    return torch.cat(resized_tensors, dim=0)




def main(args):
    # Init model
    config = Config()

    device = torch.device("cuda")
    model = GCoNet()
    model = model.to(device)
    print('Testing with model {}'.format(args.ckpt))
    gconet_dict = torch.load(args.ckpt)

    model.to(device)
    model.load_state_dict(gconet_dict)

    model.eval()

    # Edit only the root directory of your datasets and prediction directory path.
    my_testset_root_dir = 'E:/Python_Project/Color2Embed-main/data/000-599'
    my_pred_dir = os.path.join('E:/Python_Project/GCoNet_plus-paper_CN/test_result')#输出文件夹
    target_size = (256, 256)  # 指定目标大小

    return test_dataset_folder(my_testset_root_dir, my_pred_dir, args.size, device, model,target_size)


if __name__ == '__main__':
    # Parameter from command line
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model',
                        default='GCoNet',
                        type=str,
                        help="Options: '', ''")
    parser.add_argument('--size',
                        default=224,
                        type=int,
                        help='input size')
    parser.add_argument('--ckpt', default='E:/Python_Project/GCoNet_plus-paper_CN/GCoNet.pth', type=str,
                        help='model folder')
    parser.add_argument('--pred_dir', default='E:/Python_Project/GCoNet_plus-paper_CN/test_result', type=str, help='Output folder')

    args = parser.parse_args()

    result_tensor = main(args)
    #将张量转化为图片并且保存
    print(result_tensor.size())

#将张量转化为图片并显示：
    for i in range(result_tensor.size(0)):
        img_np = result_tensor[i].cpu().numpy().transpose(1, 2, 0)
        cv2.imshow('result', img_np)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

