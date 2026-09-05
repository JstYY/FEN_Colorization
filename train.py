import argparse

import os
import cv2
import numpy as np
from PIL import Image
from skimage import color, io
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
from GCoNet.test import ImageProcessor

from tqdm import tqdm
from torchvision import transforms
# from ColorEncoder import ColorEncoder
from models import ColorEncoder, ColorUNet, PixelAttention
from vgg_model import vgg19
from data.data_loader import MultiResolutionDataset
from colorization.color import ImageColorizer
from utils import tensor_lab2rgb
from torchvision.transforms import functional as TF
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.hub.api import HubApi
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
)


YOUR_ACCESS_TOKEN = 'caf03797-77ed-4ec3-919d-cd4377a5d466'

api = HubApi()
api.login(YOUR_ACCESS_TOKEN)
# 用于取消设置的环境变量，这里您可能需要根据实际情况来设定
os.environ['VLLM_USE_MODELSCOPE'] = '0'


def rgb_to_lab_tensor(input_tensor):
    """将一个批次的RGB图像张量转换为Lab张量。"""
    # input_tensor 应该是在[0, 255]范围内的 RGB 图像批次 (batch, channels, height, width)
    rgb_images = input_tensor.permute(0, 2, 3, 1)  # 调整维度为(batch, height, width, channels)
    rgb_images = rgb_images.cpu().numpy()  # 转换为NumPy数组以使用skimage
    lab_images = np.array([color.rgb2lab(rgb_image) for rgb_image in rgb_images])  # 转换每张RGB图为Lab
    lab_tensor = torch.from_numpy(lab_images).float()  # 转换回Tensor
    lab_tensor = lab_tensor.permute(0, 3, 1, 2)  # 调整回 PyTorch 维度
    # 归一化 L 通道在 [0, 100] 范围, a 和 b 通道在 [-127, 127] 范围
    lab_tensor[:, 0, :, :] = lab_tensor[:, 0, :, :] / 50.0 - 1.0  # 归一化 L 通道
    lab_tensor[:, 1:, :, :] = lab_tensor[:, 1:, :, :] / 127.0  # 归一化 a 和 b 通道
    return lab_tensor.to(input_tensor.device)  # 确保转换后的tensor在相同设备上


def mkdirss(dirpath):  # 创建文件夹
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def data_sampler(dataset, shuffle, distributed):  # 选择数据采样器，参数分别是数据集、是否打乱数据、是否分布式
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    # DistributedSampler这种采样器用于分布式训练，确保每个训练进程只会处理全局数据集的一部分，这样每个进程都在不同的数据子集上训练。
    if shuffle:
        return data.RandomSampler(dataset)
    # RandomSampler是一个用于随机采样的采样器，它将数据集中的索引打乱，然后按照打乱后的顺序进行采样。
    else:
        return data.SequentialSampler(dataset)
    # SequentialSampler是一个用于顺序采样的采样器，它按照数据集中的顺序进行采样，不进行任何打乱工作


def requires_grad(model, flag=True):
    for p in model.parameters():
        # model.parameters() 是一个生成器，它遍历模型中的所有参数。对于这些参数，函数会根据 flag 的值设置其 requires_grad 属性
        p.requires_grad = flag


# 这个函数的目的是将模型的参数的 requires_grad 属性设置为 True 或 False。这个属性表示是否需要计算梯度。如果设置为 True，那么在进行反向传播的时候，这些参数的梯度就会被计算。

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch  # 每当循环达到一个批次，它就通过yield关键字产出批次数据。yield的作用是使得sample_data成为一个生成器，在每次迭代中暂停并返回当前批次的数据。


# 这个函数的目的是从数据加载器中无限循环地加载数据。这个函数是一个生成器，它会返回数据加载器中的每一个批次数据。当数据加载器中的数据被加载完毕后，它会重新开始加载数据。
def Lab2RGB_out(img_lab):
    img_lab = img_lab.detach().cpu()
    img_l = img_lab[:, :1, :, :]
    img_ab = img_lab[:, 1:, :, :]
    # print(torch.max(img_l), torch.min(img_l))
    # print(torch.max(img_ab), torch.min(img_ab))
    img_l = img_l + 50
    pred_lab = torch.cat((img_l, img_ab), 1)[0, ...].numpy()
    # grid_lab = utils.make_grid(pred_lab, nrow=1).numpy().astype("float64")
    # print(grid_lab.shape)
    out = (np.clip(color.lab2rgb(pred_lab.transpose(1, 2, 0)), 0, 1) * 255).astype("uint8")
    return out


def RGB2Lab(inputs):
    # input [0, 255] uint8
    # out l: [0, 100], ab: [-110, 110], float32
    return color.rgb2lab(inputs)


def Normalize(inputs):
    l = inputs[:, :, 0:1]
    ab = inputs[:, :, 1:3]
    l = l - 50
    lab = np.concatenate((l, ab), 2)

    return lab.astype('float32')


def numpy2tensor(inputs):
    out = torch.from_numpy(inputs.transpose(2, 0, 1))
    return out


def tensor2numpy(inputs):
    out = inputs[0, ...].detach().cpu().numpy().transpose(1, 2, 0)
    return out


def preprocessing(inputs):  # 预处理函数
    # input: rgb, [0, 255], uint8
    img_lab = Normalize(RGB2Lab(inputs))
    img = np.array(inputs, 'float32')  # [0, 255]#这里的img是RGB图像
    img = numpy2tensor(img)
    img_lab = numpy2tensor(img_lab)
    return img.unsqueeze(0), img_lab.unsqueeze(0)


def uncenter_l(inputs):
    l = inputs[:, :1, :, :] + 50
    ab = inputs[:, 1:, :, :]
    return torch.cat((l, ab), 1)


def train(
        args,
        loader,
        colorEncoder,
        colorUNet,
        vggnet,
        g_optim,
        device,
):  # 参数分别是训练参数、数据加载器、颜色编码器、颜色UNet、VGG网络、优化器和设备。
    loader = sample_data(loader)
    # sample_data 函数返回一个生成器，它会无限循环地加载数据。这个生成器会返回数据加载器中的每一个批次数据。当数据加载器中的数据被加载完毕后，它会重新开始加载数据。
    pbar = range(args.iter)
    # pbar 是一个迭代器，它会在训练过程中产生一个从 0 到 args.iter 的整数序列。这个迭代器会在每次迭代中产生一个新的整数，这个整数表示当前的迭代次数。
    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)
    # 如果当前进程的 rank 是 0，那么就创建一个 tqdm 迭代器。这个迭代器会在训练过程中产生一个从 args.start_iter 到 args.iter 的整数序列。这个迭代器会在每次迭代中产生一个新的整数，这个整数表示当前的迭代次数。
    g_loss_val = 0  # 初始化损失值为0
    loss_dict = {}  # 初始化损失字典为空
    # 注意力模型的初始化和优化器
    pa_optim = optim.Adam(pixel_attention.parameters(), lr=args.lr, betas=(0.9, 0.99))
    if args.distributed:
        colorEncoder_module = colorEncoder.module
        colorUNet_module = colorUNet.module
    # 在分布式训练中，模型通常会被包装进一个DistributedDataParallel（DDP）容器，以支持在多个GPU上同步训练。当你需要访问原始的未经DDP包装的模型时，你需要通过.module属性来访问它。
    # 这样，不管是训练还是保存模型，你都在使用原始的colorEncoder和colorUNet模块，而不是它们的DDP包装版本。
    else:  # 对于非分布式训练，没有必要访问.module属性，因为模型没有被DDP包装，可以直接使用。
        colorEncoder_module = colorEncoder
        colorUNet_module = colorUNet
    # 如果是分布式训练，那么就将颜色编码器和颜色UNet模型的 module 属性赋值给 colorEncoder_module 和 colorUNet_module。否则，就直接将颜色编码器和颜色UNet模型赋值给 colorEncoder_module 和 colorUNet_module。
    colorizer = ImageColorizer(output_folder1="E:\Python_Project\Color2Embed-main\\test_datasets\gray2color\otest1",
                               output_folder2="E:\Python_Project\Color2Embed-main\\test_datasets\gray2color\otest2"
                               )
    processor = ImageProcessor()
    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")

            break
        # 这个for循环是迭代训练过程的主体，其中pbar是一个用于追踪训练进度的progressbar对象。
        # 在每次迭代中，它会从数据加载器中加载一个批次的数据，然后将这个批次的数据传递给模型进行训练。
        img, img_ref, img_lab = next(loader)
        # print(img.shape)
        # next(loader) 会从 loader 中加载一个批次的数据。这个批次的数据会被分别赋值给 img, img_ref 和 img_lab。
        # 这一行从数据加载器加载包含训练样本的一批数据。其中，`img`是输入图像，`img_ref`是参考图像，而`img_lab`是目标的Lab色彩空间表示。
        # img_lab是目标的Lab色彩空间表示，它包含了亮度通道L和颜色通道ab。这个表示是为了训练模型，使得模型能够将输入图像的亮度通道L转换为目标的颜色通道ab。
        # 目标的颜色通道ab是通过参考图像的Lab色彩空间表示计算得到的。
        # 在这调用模型使得img变成处理后的彩色图像

        # colorizer = ImageColorizer(output_folder="E:/Python_Project/Color2Embed-main/data/train_color")
        img_dd = colorizer.colorize(img, save_to_folder=True)

        while img_dd.shape != img.shape:
            img_dd = colorizer.colorize(img, save_to_folder=True)
            # 上色图像并保存到文件夹
            # print(img_dd.shape,img.shape)
            # 上色图像并保存到文件夹
            # print(img_dd.shape,img.shape)
        # 我们希望经过预训练的上色模型将灰度图上色后，根据参考图像的色彩信息，对彩色图像进行补充上色。
        img = img.to(device)  ## 将图像移动到指定设备（例如GPU）#这里图像的数据类型是torch.Tensor，它是PyTorch中的张量数据类型。这里将输入图像移动到指定设备（例如GPU）。
        img_lab = img_lab.to(device)  # 将参考图像移动到指定设备（例如GPU）

        img_ref = img_ref.to(device)  # tps_transformed image RGB [B, 3, 256, 256] 将目标Lab值移动到指定设备（例如GPU）

        img_l = img_lab[:, :1, :, :] / 50  # [-1, 1] target L#将输入图像的亮度通道L标准化到[-1,1]范围，颜色为灰度图像。

        img_ab = img_lab[:, 1:, :, :] / 110  # [-1, 1] target ab#
        # img_ref_ab = img_ref_lab[:,1:,:,:] / 110 # [-1, 1] ref ab
        # 这些步骤涉及将图像移动到计算设备并进行必要的缩放。Lab色彩空间的分量通常被缩放到-1到1的范围以适应模型。
        colorEncoder.train()
        colorUNet.train()

        requires_grad(colorEncoder, True)
        requires_grad(colorUNet, True)
        # 这确保了`colorEncoder`和`colorUNet`在训练模式下运行，模块中的某些层可能会根据模式（训练或评估）表现不同，如Dropout和BatchNorm。
        ref_color_vector = colorEncoder(img_ref / 255.)

        fake_swap_ab = colorUNet((img_l, ref_color_vector))

        img_dd_lab = rgb_to_lab_tensor(img_dd)
        pretrained_ab = img_dd_lab[:, 1:, :, :]  #
        mask_tensor = processor.process_images(img_dd, img_ref)
        pretrained_ab = pretrained_ab.to(fake_swap_ab.device)
        pixel_attention.train()
        fake_swap_ab = fake_swap_ab.to(device)
        pretrained_ab = pretrained_ab.to(device)
        mask_tensor = mask_tensor.to(device)
        fused_ab = pixel_attention(fake_swap_ab, pretrained_ab,mask_tensor)

        recon_loss = F.smooth_l1_loss(fused_ab, img_ab) * 1

        # 特征损失，通过VGG网络计算
        real_img_rgb = img / 255  # img是输入图像，它是一个张量，表示灰度图像。这里将其转换为RGB格式，并将其标准化到[0,1]范围。
        features_A = vggnet(real_img_rgb, layer_name='all')
        fake_swap_rgb = tensor_lab2rgb(torch.cat((img_l * 50 + 50, fused_ab * 110), 1))  # [0, 1]
        features_B = vggnet(fake_swap_rgb, layer_name='all')
        # fea_loss = F.l1_loss(features_A[-1], features_B[-1]) * 0.1
        # fea_loss = 0

        fea_loss1 = F.l1_loss(features_A[0], features_B[0]) / 32 * 0.1
        fea_loss2 = F.l1_loss(features_A[1], features_B[1]) / 16 * 0.1
        fea_loss3 = F.l1_loss(features_A[2], features_B[2]) / 8 * 0.1
        fea_loss4 = F.l1_loss(features_A[3], features_B[3]) / 4 * 0.1
        fea_loss5 = F.l1_loss(features_A[4], features_B[4]) * 0.1

        fea_loss = fea_loss1 + fea_loss2 + fea_loss3 + fea_loss4 + fea_loss5
        # 目标是使生成的图像与实际图像在Lab空间（在这里仅使用ab分量）上尽可能相似，并且在特征空间（通过VGG网络提取的特征）上也相似。
        # 这里会计算两种损失：重建损失和特征损失。
        loss_dict["recon"] = recon_loss

        loss_dict["fea"] = fea_loss
        # 只进行一次 backward pass
        g_optim.zero_grad()  # 清除之前的梯度
        (recon_loss + fea_loss).backward()  # 计算总损失的梯度
        g_optim.step()  # 根据梯度更新模型的权重
        pa_optim.zero_grad()
        pa_optim.step()
        loss_reduced = reduce_loss_dict(loss_dict)

        recon_val = loss_reduced["recon"].mean().item()
        # recon_val 是重建损失的平均值
        fea_val = loss_reduced["fea"].mean().item()
        # fea_val 是特征损失的平均值
        # 这里，我们首先从`loss_dict`中提取重建损失（`recon_loss`）和特征损失（`fea_loss`）的值，
        # 将它们通过`.mean().item()`转换为Python的标量值，
        # 并赋值给`recon_val`和`fea_val`这两个变量。这样我们便可以记录和监控模型在训练过程中的表现。
        if get_rank() == 0:  # 如果当前进程的 rank 是 0，那么就更新进度条的描述信息。
            pbar.set_description(
                (
                    f"recon:{recon_val:.4f}; fea:{fea_val:.4f};"
                )
            )

            if i % 500 == 0:
                with torch.no_grad():
                    colorEncoder.eval()
                    colorUNet.eval()  #
                    pixel_attention.eval()

                    imgsize = 256
                    for inum in range(10):
                        val_img_path = 'test_datasets/val_datasets/in%d.JPEG' % (inum + 1)
                        val_ref_path = 'test_datasets/val_datasets/ref%d.JPEG' % (inum + 1)
                        # val_img_path = 'test_datasets/val_daytime/day_sample/in%d.jpg'%(inum+1)
                        # val_ref_path = 'test_datasets/val_daytime/night_sample/dark4.jpg'
                        out_name = 'in%d_ref%d.png' % (inum + 1, inum + 1)
                        val_img = Image.open(val_img_path).convert("RGB").resize((imgsize, imgsize))
                        val_img_ref = Image.open(val_ref_path).convert("RGB").resize((imgsize, imgsize))
                        val_img, val_img_lab = preprocessing(val_img)

                        val_img_ref, val_img_ref_lab = preprocessing(val_img_ref)
                        val_img_dd = colorizer.colorize(val_img_lab, save_to_folder=True)

                        # val_img = val_img.to(device)
                        val_img_lab = val_img_lab.to(device)
                        val_img_ref = val_img_ref.to(device)
                        # val_img_ref_lab = val_img_ref_lab.to(device)

                        val_img_l = val_img_lab[:, :1, :, :] / 50
                        # val_img_ref_ab = val_img_ref_lab[:,1:,:,:] / 110. # [-1, 1]

                        ref_color_vector = colorEncoder(val_img_ref / 255.)  # [0, 1]
                        fake_swap_ab = colorUNet((val_img_l, ref_color_vector))

                        val_img_dd_lab = rgb_to_lab_tensor(val_img_dd)
                        pretrained_ab = val_img_dd_lab[:, 1:, :, :]  #

                        pretrained_ab = pretrained_ab.to(fake_swap_ab.device)
                        mask_tensor = processor.process_images(val_img_dd, val_img_ref)
                        mask_tensor = mask_tensor.to(device)


                        val_fused_ab = pixel_attention(fake_swap_ab, pretrained_ab,mask_tensor)
                        val_fused_ab_first_image = val_fused_ab[0].unsqueeze(0)  # 第一个维度是batch_size，这里是1

                        fake_img = torch.cat((val_img_l * 50, val_fused_ab_first_image * 110), 1)  #
                        fake_img = torch.narrow(fake_img, dim=1, start=0, length=3)
                        val_img = torch.narrow(val_img, dim=1, start=0, length=3)

                        sample = np.concatenate(
                            (tensor2numpy(val_img), tensor2numpy(val_img_ref), Lab2RGB_out(fake_img)),
                            1)  # 将输入图像、参考图像和生成的图像连接在一起，形成一个大的图像。

                        out_dir = 'training_logs/%s/%06d' % (args.experiment_name, i)
                        mkdirss(out_dir)
                        io.imsave('%s/%s' % (out_dir, out_name), sample.astype('uint8'))
                        torch.cuda.empty_cache()
            if i % 2500 == 0:
                out_dir = "experiments/%s" % (args.experiment_name)
                mkdirss(out_dir)
                # 存储模型参数
                torch.save(
                    {
                        "colorEncoder": colorEncoder_module.state_dict(),
                        "colorUNet": colorUNet_module.state_dict(),
                        "pixel_attention": pixel_attention.state_dict(),  # 保存 PixelAttention 状态
                        "g_optim": g_optim.state_dict(),
                        "pa_optim": pa_optim.state_dict(),  # 保存 PixelAttention 优化器状态
                        "args": args,
                    },
                    f"{out_dir}/{str(i).zfill(6)}.pt",
                )

                print("it:",i)
                print("recon_val:",recon_val)
                print("fea_val:", fea_val)


if __name__ == "__main__":
    device = "cuda"

    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets", type=str)
    parser.add_argument("--iter", type=int, default=100000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--experiment_name", type=str, default="default")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.start_iter = 0

    vggnet = vgg19(pretrained_path='E:\\Python_Project\\Color2Embed-main\\experiments\\vgg19-dcbb9e9d.pth',
                   require_grad=False)
    vggnet = vggnet.to(device)
    vggnet.eval()

    colorEncoder = ColorEncoder(color_dim=512).to(device)
    colorUNet = ColorUNet(bilinear=True).to(device)
    pixel_attention = PixelAttention().to(device)

    g_optim = optim.Adam(
        list(colorEncoder.parameters()) + list(colorUNet.parameters()),
        lr=args.lr,
        betas=(0.9, 0.99),
    )
    if args.ckpt is not None:
        print("load model:", args.ckpt)
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        colorEncoder.load_state_dict(ckpt["colorEncoder"])
        colorUNet.load_state_dict(ckpt["colorUNet"])
        pixel_attention.load_state_dict(ckpt["pixel_attention"])  # 加载 PixelAttention 状态
        g_optim.load_state_dict(ckpt["g_optim"])
        pa_optim.load_state_dict(ckpt["pa_optim"])  # 加载 PixelAttention 优化器状态

    # print(args.distributed)

    if args.distributed:
        colorEncoder = nn.parallel.DistributedDataParallel(
            colorEncoder,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        colorUNet = nn.parallel.DistributedDataParallel(
            colorUNet,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=(0, 360))
        ]
    )

    datasets = []
    dataset = MultiResolutionDataset(
        path='E:\Python_Project\Color2Embed-main\data\Python_ProjectColor2Embed-maindatatrain_dataImageNet_train_lmdb',
        transform=transform, resolution=256)
    datasets.append(dataset)

    loader = data.DataLoader(
        data.ConcatDataset(datasets),
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    train(
        args,
        loader,
        colorEncoder,
        colorUNet,
        vggnet,
        g_optim,
        device,
    )

