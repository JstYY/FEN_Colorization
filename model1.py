import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vgg_model import vgg19


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x


class ResBlock(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bottle_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.double_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.bottle_conv(x)
        x = self.double_conv(x) + x
        return x / math.sqrt(2)


class Down(nn.Module):
    """Downscaling with stride conv then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 4, 2, 1),
            nn.LeakyReLU(0.1, True),
            # DoubleConv(in_channels, out_channels)
            ResBlock(in_channels, out_channels)
        )

    def forward(self, x):
        x = self.main(x)

        return x

class SDFT(nn.Module):  # 这个类是用来实现全局调整的，在整个网络中，对特征图进行调整，使得特征图的分布更加符合输入的颜色向量

    def __init__(self, color_dim, channels, kernel_size=3):
        super().__init__()

        # generate global conv weights
        fan_in = channels * kernel_size ** 2
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.scale = 1 / math.sqrt(fan_in)
        self.modulation = nn.Conv2d(color_dim, channels, 1)
        self.weight = nn.Parameter(
            torch.randn(1, channels, channels, kernel_size, kernel_size)
        )

    def forward(self, fea, color_style):
        # for global adjustation
        B, C, H, W = fea.size()
        # print(fea.shape, color_style.shape)
        style = self.modulation(color_style).view(B, 1, C, 1, 1)
        weight = self.scale * self.weight * style
        # demodulation
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
        weight = weight * demod.view(B, C, 1, 1, 1)

        weight = weight.view(
            B * C, C, self.kernel_size, self.kernel_size
        )

        fea = fea.view(1, B * C, H, W)
        fea = F.conv2d(fea, weight, padding=self.padding, groups=B)
        fea = fea.view(B, C, H, W)

        return fea


class UpBlock(nn.Module):

    def __init__(self, color_dim, in_channels, out_channels, kernel_size=3, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(in_channels // 2 + in_channels // 8, out_channels, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True)
        )

        self.conv_s = nn.Conv2d(in_channels // 2, out_channels, 1, 1, 0)

        # generate global conv weights
        self.SDFT = SDFT(color_dim, out_channels, kernel_size)

    def forward(self, x1, x2, color_style):
        # print(x1.shape, x2.shape, color_style.shape)
        x1 = self.up(x1)
        x1_s = self.conv_s(x1)

        x = torch.cat([x1, x2[:, ::4, :, :]], dim=1)
        x = self.conv_cat(x)
        x = self.SDFT(x, color_style)

        x = x + x1_s

        return x

class ColorEncoder(nn.Module):
    def __init__(self, color_dim=512):
        super(ColorEncoder, self).__init__()

        # self.vgg = vgg19(pretrained_path=None)
        self.vgg = vgg19()

        self.feature2vector = nn.Sequential(
            nn.Conv2d(color_dim, color_dim, 4, 2, 2), # 8x8
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(color_dim, color_dim, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(color_dim, color_dim, 4, 2, 2), # 4x4
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(color_dim, color_dim, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool2d((1, 1)), # 1x1
            nn.Conv2d(color_dim, color_dim//2, 1), # linear-1
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(color_dim//2, color_dim//2, 1), # linear-2
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(color_dim//2, color_dim, 1), # linear-3
        )

        self.color_dim = color_dim

    def forward(self, x):
        # x #[0, 1] RGB
        vgg_fea = self.vgg(x, layer_name='relu5_2') # [B, 512, 16, 16]

        x_color = self.feature2vector(vgg_fea[-1]) # [B, 512, 1, 1]

        return x_color



class ColorUNet(
    nn.Module):  # PyTorch 框架中，通过构建一个类继承自 nn.Module 基类来创建神经网络模型。在这个类里，你定义了网络的层（layers）和前向传播（forward pass）的过程。当你创建了这个类的一个实例后，你就可以向普通的函数一样调用它来进行前向传播，底层的 PyTorch 会自动处理许多事情，例如梯度的计算等。
    # UNet架构由编码器和解码器两部分组成，编码器用于逐渐捕捉输入图像的上下文信息，而解码器则用于将编码器的特征映射上采样为与输入图像相同分辨率的预测结果。
    ### this model output is ab
    # 基于UNet架构的神经网络模型，用于将灰度图像转换为彩色图像，并输出ab通道
    # ab通道是指色度通道，用于表示图像的色彩信息，由三个通道表示，分别是L（亮度）、a（颜色对立分量）和b（颜色对立分量）。L通道表示图像的亮度信息，a和b通道表示图像的色彩信息。
    def __init__(self, n_channels=1, n_classes=3, bilinear=True):
        # 构造函数 __init__ 负责初始化网络中的各个层，而 forward 方法则定义了这些层如何按顺序执行来完成前向传播。这是由 PyTorch 的设计规定的。

        # n_channels表示输入图像的通道数，默认为1，
        # n_classes表示输出图像的通道数，默认为3，
        # bilinear表示是否使用双线性插值进行上采样，默认为True。双线性插值相对于普通的上采样方法更加精确和平滑，能够提供更好的图像细节和视觉效果。然而，双线性插值的计算成本较高，可能会导致一定的运算开销。
        # 上采样（Upsampling）是将图像的尺寸增大的过程，通常通过插值算法将像素值从输入图像中的已知位置添加到新的位置上，从而获得更高分辨率的图像。上采样可以增加图像细节并提高分辨率。
        # 下采样（Downsampling）是将图像的尺寸减小的过程，通常通过降低采样率来实现。下采样可以减少图像的尺寸和信息量，适用于图像压缩和降噪等任务。
        super(ColorUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        # 编码器部分由inc、down1、down2、down3和down4组成，解码器部分由up1、up2、up3、up4和outc组成
        # 在卷积神经网络中进行下采样的主要目的是逐渐提取更加高级别的特征，并减少计算量。当减少图像的空间维度（高度和宽度）时，我们希望在剩余的特征中捕捉到更多信息。
        # 因此，增加通道数来保留更多特征是有用的。更深的网络层不仅仅看局部特征（如边缘和角点），而是能够判断形状、纹理等更高级的概念。增加通道数意味着网络能够学习更多种类的这些高级抽象特征。
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = UpBlock(512, 1024, 512 // factor, 3, bilinear)
        self.up2 = UpBlock(512, 512, 256 // factor, 3, bilinear)
        self.up3 = UpBlock(512, 256, 128 // factor, 5, bilinear)
        self.up4 = UpBlock(512, 128, 64, 5, bilinear)
        self.outc = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 2, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):  #
        # print(torch.max(x[0]), torch.min(x[0])) #[-1, 1] gray image L
        # print(torch.max(x[1]), torch.min(x[1])) # color vector

        x_color = x[1]  # [B, 512, 1, 1]#x
        # print(x[0].shape, x_color.shape)#x[0]在哪里读入的？回答：在prepare_data.py中读入的，具体是
        #x[1]是在哪里读入的？回答：在prepare_data.py中读入的，具体是在prepare_data.py中的resize_worker函数中读入的
        # 输入图像的色彩向量提取出来，作为特征向量x_color
        #x[0]是灰度图像，x[1]是色彩向量，哪一个是参考图像，哪一个是待转换图像？回答：x[0]是参考图像，x[1]是待转换图像


        x1 = self.inc(x[0])  # [B, 64, 256, 256]
        x2 = self.down1(x1)  # [B, 128, 128, 128]
        x3 = self.down2(x2)  # [B, 256, 64, 64]
        x4 = self.down3(x3)  # [B, 512, 32, 32]
        x5 = self.down4(x4)  # [B, 512, 16, 16]

        x6 = self.up1(x5, x4, x_color)  # [B, 256, 32, 32]
        x7 = self.up2(x6, x3, x_color)  # [B, 128, 64, 64]
        x8 = self.up3(x7, x2, x_color)  # [B, 64, 128, 128]
        x9 = self.up4(x8, x1, x_color)  # [B, 64, 256, 256]
        x_ab = self.outc(x9)

        return x_ab


class SomeSemanticSegmentationNet(nn.Module):
    # 这里用一个极其简化的模拟语义分割网络来表示
    def __init__(self):
        super(SomeSemanticSegmentationNet, self).__init__()
        # 假设这个模型有一层卷积
        self.conv = nn.Conv2d(3, 10, kernel_size=3, stride=1)

    def forward(self, x):
        return torch.sigmoid(self.conv(x))  # 一个简化的语义图


class SemanticGuidedAttention(nn.Module):
    def __init__(self, in_channels):
        super(SemanticGuidedAttention, self).__init__()
        self.semantic_segmentation = SomeSemanticSegmentationNet()
        self.conv_diff = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1,
                                   padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, model_output_ab, pretrained_output_ab, target_gray, ref_color):
        # 分别对目标灰度图像和参考彩色图像进行语义分割
        target_semantics = self.semantic_segmentation(target_gray)
        ref_semantics = self.semantic_segmentation(ref_color)

        # 计算语义相似度
        semantic_similarity = self.calculate_similarity(target_semantics, ref_semantics)

        # 计算两个ab通道的差异
        ab_diff = torch.abs(model_output_ab - pretrained_output_ab)

        # 获取差异权重
        diff_weight = self.sigmoid(self.conv_diff(ab_diff))

        # 根据差异权重和语义相似度调整注意力
        attention_weights = self.sigmoid(diff_weight) * semantic_similarity
        fused_output_ab = model_output_ab * attention_weights + pretrained_output_ab * (1 - attention_weights)

        return fused_output_ab

    def calculate_similarity(self, target_semantics, ref_semantics):
        # 根据实际情况定义计算语义相似度的方法
        # 这里用简单的欧氏距离的倒数作为相似度,实际应用中需要更加复杂的计算
        dist = torch.norm(target_semantics - ref_semantics, dim=1, keepdim=True)
        return 1 / (1 + dist)


# 接下来是使用新的注意力机制模块的网络定义
class ModifiedColorUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, bilinear=True):
        super(ModifiedColorUNet, self).__init__()
        # 先初始化已有的ColorUNet模块
        self.color_unet = ColorUNet(n_channels, n_classes, bilinear)
        # 初始化SemanticGuidedAttention模块
        self.semantic_guided_attention = SemanticGuidedAttention(n_classes)

    def forward(self, target_gray, ref_color, pretrained_output_ab):
        # 使用ColorUNet得到上色结果
        model_output_ab = self.color_unet(target_gray, ref_color)
        # 使用SemanticGuidedAttention得到语义引导的注意力融合后的结果
        guided_output_ab = self.semantic_guided_attention(
            model_output_ab, pretrained_output_ab, target_gray, ref_color
        )
        return guided_output_ab

class PixelAttention(nn.Module):
    def __init__(self):
        super(PixelAttention, self).__init__()
        # 假设输入是两个[N, 2, H, W]大小的ab通道堆叠，共四个通道
        self.conv_diff = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, model_output_ab, pretrained_output_ab):
        # 计算两个ab通道的差异
        ab_diff = torch.abs(model_output_ab - pretrained_output_ab)

        # 获取差异权重
        diff_weight = self.sigmoid(self.conv_diff(ab_diff))

        # 根据差异权重将注意力分配到两个ab通道
        attention_weights = self.sigmoid(diff_weight)  # [N, 2, H, W]
        fused_output_ab = model_output_ab * attention_weights + pretrained_output_ab * (1 - attention_weights)

        return fused_output_ab
