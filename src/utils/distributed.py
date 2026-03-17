"""
分布式训练工具函数

提供 PyTorch DDP 相关的工具函数，包括：
- 分布式环境初始化和清理
- 进程信息获取（rank, world_size）
- 进程间通信（barrier, all_reduce, all_gather）
"""

import os
import torch
import torch.distributed as dist
from typing import Optional, List, Dict, Any


def is_distributed() -> bool:
    """
    检查是否处于分布式环境

    Returns:
        True 如果已初始化分布式环境
    """
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """
    获取进程总数

    Returns:
        分布式环境中的进程数，非分布式环境返回 1
    """
    if is_distributed():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """
    获取当前进程的全局 rank

    Returns:
        当前进程的 rank（0 到 world_size-1），非分布式环境返回 0
    """
    if is_distributed():
        return dist.get_rank()
    return 0


def get_local_rank() -> int:
    """
    获取当前节点内的 local rank

    用于设置 CUDA 设备。

    Returns:
        当前节点内的 GPU 索引
    """
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    """
    检查是否为主进程（rank 0）

    用于控制日志打印、检查点保存等只应在主进程执行的操作。

    Returns:
        True 如果是主进程
    """
    return get_rank() == 0


def barrier():
    """
    同步所有进程

    阻塞直到所有进程都到达此点。
    非分布式环境下不做任何操作。
    """
    if is_distributed():
        dist.barrier()


def all_reduce(tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """
    全局归约操作

    对所有进程的 tensor 进行归约操作（如求和、求平均）。

    Args:
        tensor: 要归约的张量（原地修改）
        op: 归约操作，默认 SUM

    Returns:
        归约后的张量
    """
    if is_distributed():
        dist.all_reduce(tensor, op=op)
    return tensor


def all_gather(tensor: torch.Tensor) -> List[torch.Tensor]:
    """
    全局收集

    收集所有进程的 tensor。

    Args:
        tensor: 要收集的张量

    Returns:
        所有进程的张量列表
    """
    if not is_distributed():
        return [tensor]

    gathered = [torch.zeros_like(tensor) for _ in range(get_world_size())]
    dist.all_gather(gathered, tensor)
    return gathered


def all_gather_object(obj: Any) -> List[Any]:
    """
    全局收集任意 Python 对象

    Args:
        obj: 要收集的对象

    Returns:
        所有进程的对象列表
    """
    if not is_distributed():
        return [obj]

    gathered = [None for _ in range(get_world_size())]
    dist.all_gather_object(gathered, obj)
    return gathered


def reduce_dict(input_dict: Dict[str, torch.Tensor], average: bool = True) -> Dict[str, torch.Tensor]:
    """
    归约字典中的所有值

    Args:
        input_dict: 输入字典，值为 tensor
        average: 是否取平均

    Returns:
        归约后的字典
    """
    if not is_distributed():
        return input_dict

    names = sorted(input_dict.keys())
    values = torch.stack([input_dict[k] for k in names])

    all_reduce(values)
    if average:
        values /= get_world_size()

    return {k: v for k, v in zip(names, values)}


def broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """
    广播张量

    从源进程广播张量到所有进程。

    Args:
        tensor: 要广播的张量
        src: 源进程 rank

    Returns:
        广播后的张量
    """
    if is_distributed():
        dist.broadcast(tensor, src=src)
    return tensor


def init_distributed(backend: str = "nccl") -> bool:
    """
    初始化分布式环境

    从环境变量读取 RANK、WORLD_SIZE 等信息。
    通常由 torchrun 自动设置这些环境变量。

    Args:
        backend: 通信后端，默认 nccl（适用于 GPU）
                 - nccl: GPU 推荐
                 - gloo: CPU 或 GPU
                 - mpi: 需要 MPI 支持

    Returns:
        True 如果成功初始化，False 如果不需要初始化或已初始化
    """
    if is_distributed():
        return True

    # 从环境变量获取信息
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        return True

    return False


def cleanup_distributed():
    """
    清理分布式环境

    训练结束时调用，释放分布式资源。
    """
    if is_distributed():
        dist.destroy_process_group()


def setup_ddp(
    rank: int = None,
    world_size: int = None,
    backend: str = "nccl",
    init_method: str = "env://",
    port: int = None,
) -> bool:
    """
    手动设置分布式环境

    用于不使用 torchrun 的情况。

    Args:
        rank: 当前进程 rank
        world_size: 进程总数
        backend: 通信后端
        init_method: 初始化方法，可以是 "env://" 或 "tcp://..."
        port: 用于通信的端口号

    Returns:
        True 如果成功初始化
    """
    if rank is None:
        rank = get_rank()
    if world_size is None:
        world_size = get_world_size()

    if world_size <= 1:
        return False

    if is_distributed():
        return True

    # 设置端口
    if port is not None:
        os.environ["MASTER_PORT"] = str(port)

    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size
    )
    return True


class DistributedContext:
    """
    分布式上下文管理器

    用于临时设置分布式环境，退出时自动清理。

    Usage:
        with DistributedContext(rank=0, world_size=2):
            # 分布式训练代码
            pass
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        backend: str = "nccl",
        init_method: str = "env://"
    ):
        self.rank = rank
        self.world_size = world_size
        self.backend = backend
        self.init_method = init_method
        self._was_initialized = False

    def __enter__(self):
        self._was_initialized = is_distributed()
        if not self._was_initialized:
            setup_ddp(self.rank, self.world_size, self.backend, self.init_method)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._was_initialized:
            cleanup_distributed()
        return False