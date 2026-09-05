from io import BytesIO

import numpy as np
import lmdb
from PIL import Image
from skimage import color
import torch
from torch.utils.data import Dataset
from data.tps_transformation import tps_transform

def RGB2Lab(inputs):
    return color.rgb2lab(inputs)

def Normalize(inputs):
    # output l [-50,50] ab[-128,128]#颜色通道ab在这个函数中没有被改变，这意味着通道ab保持原始输入范围[-128,128]
    l = inputs[:, :, 0:1]#截取亮度通道L，这将提取数组中第三个维度的第0个通道。
    ab = inputs[:, :, 1:3]#截取颜色通道a和b。
    l = l - 50#亮度通道L的范围是[0,100]，这里将其转换为[-50,50]。因为神经网络通常对以0为中心的数据响应更好。
    # ab = ab
    lab = np.concatenate((l, ab), 2)#将标准化后的L通道和原始的颜色通道ab沿第三个维度（通道维度）合并回一个单一的数组 lab。

    return lab.astype('float32')

def selfnormalize(inputs):
    d = torch.max(inputs) - torch.min(inputs)
    out = (inputs) / d
    return out

#Normalize 函数是为了处理Lab颜色空间的图像数据，特别关注于亮度分量的平移调整；而 selfnormalize 是对任意数值数据进行标准化到[0, 1]范围内。
#这两个函数的目的是为了将输入数据转换为神经网络更容易处理的形式，这样可以提高训练的稳定性和收敛速度。
def to_gray(inputs):
    img_gray = np.clip((np.concatenate((inputs[:,:,:1], inputs[:,:,:1], inputs[:,:,:1]), 2)+50)/100*255, 0, 255).astype('uint8')
    
    return img_gray

def numpy2tensor(inputs):
    out = torch.from_numpy(inputs.transpose(2,0,1))
    return out

class MultiResolutionDataset(Dataset):#这个类作用是将图像数据从 LMDB 数据库中读取出来，并且对图像数据进行预处理封装。
    def __init__(self, path, transform, resolution=256):#三个参数是数据集的路径、数据集的转换函数和数据集的分辨率。
        self.env = lmdb.open(
            path ,
            max_readers=32,#允许的最大读取者数量，即同时读取数据库的进程数。
            readonly=True,#只读模式
            lock=False,#不加锁
            readahead=False,#是否预读取数据库文件的内容到内存中。对于随机读取而言，预读往往是没有必要的，并且可能会降低性能。
            meminit=False,#是否初始化内存。如果设置为True，那么在打开数据库时，将会初始化内存。
        )
        self.color_img = None#用于存储彩色图像数据。


        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:#txn是一个事务对象，用于对数据库进行读写操作。
            #begin是lmdb的一个方法，用于创建一个事务对象。事务对象是lmdb的一个核心概念，它是一组数据库操作的集合，要么全部成功，要么全部失败。
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):#魔术方法__getitem__用于获取数据集中的一个样本。在这个方法中，我们首先从数据库中获取图像数据，然后对图像数据进行预处理，最后返回处理后的图像数据。
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            #这行创建了一个键，该键格式化为字符串，由数据集的分辨率 (self.resolution) 和零填充的索引 (index) 组成，索引被转换成长度为5的字符串(zfill(5))。然后这个字符串被编码为 UTF-8 字节串，因为 LMDB 使用字节串作为键。
            img_bytes = txn.get(key)
            #这行从数据库中获取了键对应的值，即图像数据。这个值是一个字节串，我们需要将其解码为图像数据。

        buffer = BytesIO(img_bytes)
        #这行创建了一个 BytesIO 对象，用于在内存中存储图像数据。BytesIO 是 Python 的一个内置类，它提供了一个类文件对象的接口，可以用于在内存中读写二进制数据。
        img = Image.open(buffer)
        #这行创建了一个 Image 对象，用于表示图像数据。Image 是 PIL 库的一个类，它提供了一系列用于处理图像数据的方法。
        img_src = np.array(img) # [0,255
        # ] uint8
        #将 img 图像对象转换成一个 NumPy 数组 img_src。通常这个数组的数据类型为 uint8，范围从 0 到 255。

        ## add gaussian noise
        noise = np.random.uniform(-5, 5, np.shape(img_src))#uinform是一个均匀分布的随机数生成器，它生成一个形状和 img_src 相同，值介于 -5 到 5 之间的随机矩阵。它将被用来添加到图像上产生高斯噪声效果。
        #生成一个形状和 img_src 相同，值介于 -5 到 5 之间的随机矩阵。它将被用来添加到图像上产生高斯噪声效果。
        img_ref = np.clip(np.array(img_src) + noise, 0, 255)
        # 将源图像 `img_src` 数组和生成的随机高斯噪声 `noise` 相加，使用 `np.clip` 确保加噪后的像素值仍然位于合法的图像数据范围（0至255）内，得到含噪声的图像 `img_ref`。
        img_ref = tps_transform(img_ref) # [0,255] uint8# 接下来，这行代码将含噪声的图像 `img_ref` 传递给 `tps_transform` 方法，该方法将对图像应用薄板样条变形。

        img_ref = np.clip(img_ref, 0, 255)
        # 对 TPS 变换后的图像进行裁剪，以确保其像素值仍位于合法范围内。
        img_ref = img_ref.astype('uint8')## 将 `img_ref` 图像的数据类型转换为 `uint8`，这是图像处理中常见的数据类型，用于表示一个范围在0-255的整数值。
        img_ref = Image.fromarray(img_ref)## 通过 `Image.fromarray` 创建一个新的图像对象 `img_ref`，以便可以进行进一步的图像处理或保存。
        img_ref = np.array(self.transform(img_ref)) # [0,255] uint8
        ## 将转换后的图像 `img_ref` 再次转换成 NumPy 数组，并且对其应用 `self.transform`，这通常是一系列图像变换操作（如裁剪、缩放、归一化等）。
        img_lab = Normalize(RGB2Lab(img_src)) # l [-50,50] ab [-128, 128]
        ## `img_src` 从 RGB 色彩空间转换到 Lab 色彩空间，并对得到的值进行归一化处理以适应特定范围。
        img = img_src.astype('float32') # [0,255] float32 RGB
        img_ref = img_ref.astype('float32') # [0,255] float32 RGB

        img = numpy2tensor(img)
        img_ref = numpy2tensor(img_ref) # [B, 3, 256, 256]
        img_lab = numpy2tensor(img_lab)

        return img, img_ref, img_lab
        
    