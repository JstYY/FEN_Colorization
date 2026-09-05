import math
import pickle

import torch
from torch import distributed as dist
from torch.utils.data.sampler import Sampler

#这些函数总体上目的是简化和实施在多GPU或多节点环境下进行分布式训练时的数据传输和同步。这在大规模并行训练深度学习模型时尤为重要
def get_rank():#返回当前进程在分布式训练中的排名。如果没有初始化分布式环境或者不可用，它将返回0。
    if not dist.is_available():
        return 0

    if not dist.is_initialized():
        return 0

    return dist.get_rank()


def synchronize():#在所有进程中同步点。这个函数将等待直到所有进程达到这个点才继续向下执行。它在一个有多个进程的分布式环境中特别有用，可以确保所有进程在继续执行前处于同步状态。
    if not dist.is_available():
        return

    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()

    if world_size == 1:
        return

    dist.barrier()


def get_world_size():#返回分布式环境中的进程数量。如果没有初始化分布式环境或者不可用，它将返回1。
    if not dist.is_available():
        return 1

    if not dist.is_initialized():
        return 1

    return dist.get_world_size()


def reduce_sum(tensor):#对张量进行全局求和。这个函数将张量的值从所有进程中收集起来，然后对它们进行求和，然后将结果广播到所有进程中。
    if not dist.is_available():
        return tensor

    if not dist.is_initialized():
        return tensor

    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    return tensor


def gather_grad(params):#收集梯度。这个函数将梯度从所有进程中收集起来，然后对它们进行求和，然后将结果广播到所有进程中。
    world_size = get_world_size()
    
    if world_size == 1:
        return

    for param in params:
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data.div_(world_size)


def all_gather(data):#在所有进程中收集数据。这个函数将数据从所有进程中收集起来，然后将它们广播到所有进程中。
    world_size = get_world_size()

    if world_size == 1:
        return [data]

    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to('cuda')

    local_size = torch.IntTensor([tensor.numel()]).to('cuda')
    size_list = [torch.IntTensor([0]).to('cuda') for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to('cuda'))

    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to('cuda')
        tensor = torch.cat((tensor, padding), 0)

    dist.all_gather(tensor_list, tensor)

    data_list = []

    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_loss_dict(loss_dict):#对损失字典进行全局求和。这个函数将损失字典的值从所有进程中收集起来，然后对它们进行求和，然后将结果广播到所有进程中。
    world_size = get_world_size()

    if world_size < 2:
        return loss_dict

    with torch.no_grad():
        keys = []
        losses = []

        for k in sorted(loss_dict.keys()):
            keys.append(k)
            losses.append(loss_dict[k])

        losses = torch.stack(losses, 0)
        dist.reduce(losses, dst=0)

        if dist.get_rank() == 0:
            losses /= world_size

        reduced_losses = {k: v for k, v in zip(keys, losses)}

    return reduced_losses
