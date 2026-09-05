import os
import numpy as np
from skimage import color, io

import torch
import torch.nn.functional as F
from GCoNet.test import ImageProcessor

from PIL import Image
from models import ColorEncoder, ColorUNet,PixelAttention
from pytorch_fid import fid_score
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from colorization.color import ImageColorizer
def rgb_to_lab_tensor(input_tensor):
    """将一个批次的RGB图像张量转换为Lab张量。"""
    # input_tensor 应该是在[0, 255]范围内的 RGB 图像批次 (batch, channels, height, width)
    rgb_images = input_tensor.permute(0, 2, 3, 1)  # 调整维度为(batch, height, width, channels)
    rgb_images = rgb_images.cpu().numpy()  # 转换为NumPy数组以使用skimage
    lab_images = np.array([color.rgb2lab(rgb_image) for rgb_image in rgb_images])  # 转换每张RGB图为Lab
    lab_tensor = torch.from_numpy(lab_images).float()  # 转换回Tensor
    lab_tensor = lab_tensor.permute(0, 3, 1, 2)  # 调整回 PyTorch 维度
    # 归一化 L 通道在 [0, 100] 范围, a 和 b 通道在 [-127, 127] 范围
    lab_tensor[:,0,:,:] = lab_tensor[:,0,:,:] / 50.0 - 1.0  # 归一化 L 通道
    lab_tensor[:,1:,:,:] = lab_tensor[:,1:,:,:] / 127.0  # 归一化 a 和 b 通道
    return lab_tensor.to(input_tensor.device)  # 确保转换后的tensor在相同设备上

def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def Lab2RGB_out(img_lab):#这个函数是用来将Lab图像转换为RGB图像的
    img_lab = img_lab.detach().cpu()
    img_l = img_lab[:,:1,:,:]#
    img_ab = img_lab[:,1:,:,:]
    # print(torch.max(img_l), torch.min(img_l))
    # print(torch.max(img_ab), torch.min(img_ab))
    img_l = img_l + 50
    pred_lab = torch.cat((img_l, img_ab), 1)[0,...].numpy()
    # grid_lab = utils.make_grid(pred_lab, nrow=1).numpy().astype("float64")
    # print(grid_lab.shape)
    out = (np.clip(color.lab2rgb(pred_lab.transpose(1, 2, 0)), 0, 1)* 255).astype("uint8")
    return out

def RGB2Lab(inputs):
    return color.rgb2lab(inputs)

def Normalize(inputs):
    l = inputs[:, :, 0:1]
    ab = inputs[:, :, 1:3]
    l = l - 50
    lab = np.concatenate((l, ab), 2)

    return lab.astype('float32')

def numpy2tensor(inputs):
    out = torch.from_numpy(inputs.transpose(2,0,1))
    return out

def tensor2numpy(inputs):
    out = inputs[0,...].detach().cpu().numpy().transpose(1,2,0)
    return out

def preprocessing(inputs):
    # input: rgb, [0, 255], uint8
    img_lab = Normalize(RGB2Lab(inputs))
    img = np.array(inputs, 'float32') # [0, 255]
    img = numpy2tensor(img)#这里的img是RGB图像
    img_lab = numpy2tensor(img_lab)
    return img.unsqueeze(0), img_lab.unsqueeze(0)

if __name__ == "__main__":
    device = "cuda"
    # 使用ImageColorizer上色，保存到文件夹，并将结果转换为Lab格式
    colorizer = ImageColorizer(
        output_folder1="E:\Python_Project\Color2Embed-main\\test_datasets\gray2color\otest1",
        output_folder2="E:\Python_Project\Color2Embed-main\\test_datasets\gray2color\otest2")
    model_name = 'Color2Embed_2_5.50w_con'
    processor = ImageProcessor()
    ckpt_path = 'E:/Python_Project/Color2Embed-main/experiments/Color2Embed_1/055000.pt'
    test_dir_path = 'E:\Python_Project\Color2Embed-main\\test_datasets\contact'
    out_dir_path = 'results/gray2color/exemplar_based/' + model_name
    imgs_num = len(os.listdir(test_dir_path)) // 2
    imgsize = 256

    mkdirs(out_dir_path)

    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)

    colorEncoder = ColorEncoder().to(device)
    colorEncoder.load_state_dict(ckpt["colorEncoder"])
    colorEncoder.eval()

    colorUNet = ColorUNet().to(device)
    colorUNet.load_state_dict(ckpt["colorUNet"])
    colorUNet.eval()

    # pixel_attention = PixelAttention().to(device)
    # pixel_attention.load_state_dict(ckpt["pixel_attention"])
    # pixel_attention.eval()
    imgs = []
    imgs_lab = []
    cos_d_avg = 0

    for i in range(imgs_num):
        idx = i
        print('Image', idx, 'Input Image', 'in%d.JPEG'%idx, 'Ref Image', 'ref%d.JPEG'%idx)

        img_path = os.path.join(test_dir_path, '%03d_in.jpg'%idx)
        ref_img_path = os.path.join(test_dir_path, '%03d_ref.jpg'%idx)
        out_img_path = os.path.join(out_dir_path, '%03d_in.png'%idx)

        img1 = Image.open(img_path).convert("RGB")#这里的img1是灰度图像
        width, height = img1.size#这里的width和height是灰度图像的宽和高
        img2 = Image.open(ref_img_path).convert("RGB")#这里的img2是彩色图像

        img1, img1_lab = preprocessing(img1)#这里的img1是灰度图像
        img2, img2_lab = preprocessing(img2)#这里的img2是彩色图像

        img1 = img1.to(device)#
        img1_lab = img1_lab.to(device)
        img2 = img2.to(device)
        img2_lab = img2_lab.to(device)

        # print('-------',torch.max(img1_lab[:,:1,:,:]), torch.min(img1_lab[:,1:,:,:]))

        with torch.no_grad():
            # 缩放引用图像并且计算color_vector
            img2_resize = F.interpolate(img2 / 255., size=(imgsize, imgsize), mode='bilinear',
                                        recompute_scale_factor=False, align_corners=False)
            color_vector = colorEncoder(img2_resize)

            # 缩放并计算灰度图像的fake_ab通道
            img1_L_resize = F.interpolate(img1_lab[:, :1, :, :] / 50., size=(imgsize, imgsize), mode='bilinear',
                                          recompute_scale_factor=False, align_corners=False)
            fake_ab = colorUNet((img1_L_resize, color_vector))


            img_dd = colorizer.colorize(img1, save_to_folder=True)
            img_dd_rgb = img_dd / 255.  # 假设img_dd是[0,255]范围的RGB图像
            img_dd_lab = rgb_to_lab_tensor(img_dd_rgb)  # 获取转换后的Lab图像张量
            pretrained_ab = img_dd_lab[:, 1:, :, :]  # 提取ab通道

            # 进行插值以匹配fake_ab大小
            pretrained_ab_resized = F.interpolate(pretrained_ab, size=fake_ab.shape[2:], mode='bilinear',
                                                  align_corners=False)

            # 将预训练模型的输出移动到GPU上 (确保fake_swap_ab已经在这里定义了)
            pretrained_ab_resized = pretrained_ab_resized.to(fake_ab.device)
            mask_tensor = processor.process_images(img_dd, color_vector)
            # 注意力机制融合ab通道
            pixel_attention.eval()  # 将注意力机制模块设置为评估模式
            fused_ab = pixel_attention(fake_ab * 110, pretrained_ab_resized * 110,)  # 融合
            fused_ab = F.interpolate(fused_ab / 110, size=(height, width), mode='bilinear', align_corners=False)

            # 将L通道与融合后的ab通道结合，并进行Lab到RGB的转换
            fake_img_lab = torch.cat((img1_lab[:, :1, :, :], fused_ab), 1)
            fake_img_rgb = Lab2RGB_out(fake_img_lab)

            # 保存生成的图像
            io.imsave(out_img_path, fake_img_rgb)#这里的fake_img是RGB图像

            re_img, re_img_lab = preprocessing(fake_img)
            re_img = re_img.to(device)
            re_img_resize = F.interpolate(re_img / 255., size=(imgsize, imgsize), mode='bilinear', recompute_scale_factor=False, align_corners=False)
            re_color_vector = colorEncoder(re_img_resize)

            cos_d = torch.cosine_similarity(color_vector, re_color_vector, dim=1)
            cos_d_avg += cos_d

    print('Average Cosine Distance is: ', cos_d_avg/imgs_num)


