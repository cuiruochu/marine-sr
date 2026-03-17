"""
分布式训练测试

注意：这些测试需要多 GPU 环境才能完整运行。
单卡环境下会跳过大部分测试。

运行方式：
    # 单卡测试
    uv run pytest tests/test_distributed.py -v

    # 多卡测试（需要 torchrun）
    torchrun --nproc_per_node=2 -m pytest tests/test_distributed.py -v -k "ddp"
"""

import pytest
import torch
import torch.distributed as dist
import os
import tempfile
from pathlib import Path

from src.utils.distributed import (
    is_distributed,
    get_rank,
    get_world_size,
    get_local_rank,
    is_main_process,
    barrier,
    all_reduce,
    all_gather,
    init_distributed,
    cleanup_distributed,
)


class TestDistributedUtils:
    """分布式工具函数测试（单卡环境）"""

    def test_is_distributed_single_process(self):
        """单进程环境下测试 is_distributed"""
        # 如果没有初始化分布式，应该返回 False
        if not dist.is_initialized():
            assert is_distributed() == False

    def test_get_rank_single_process(self):
        """单进程环境下测试 get_rank"""
        if not dist.is_initialized():
            assert get_rank() == 0

    def test_get_world_size_single_process(self):
        """单进程环境下测试 get_world_size"""
        if not dist.is_initialized():
            assert get_world_size() == 1

    def test_get_local_rank_default(self):
        """测试 get_local_rank 默认值"""
        # 默认应该返回 0
        assert get_local_rank() >= 0

    def test_is_main_process_single_process(self):
        """单进程环境下测试 is_main_process"""
        if not dist.is_initialized():
            assert is_main_process() == True

    def test_barrier_single_process(self):
        """单进程环境下测试 barrier（不应该阻塞）"""
        # 应该正常通过，不阻塞
        barrier()

    def test_all_reduce_single_process(self):
        """单进程环境下测试 all_reduce"""
        if dist.is_initialized():
            pytest.skip("分布式环境已初始化，跳过单进程测试")

        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = all_reduce(tensor.clone())
        # 单进程模式下，结果应该不变
        assert torch.allclose(result, tensor)

    def test_all_gather_single_process(self):
        """单进程环境下测试 all_gather"""
        if dist.is_initialized():
            pytest.skip("分布式环境已初始化，跳过单进程测试")

        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = all_gather(tensor.clone())
        # 单进程模式下，应该返回只有一个元素的列表
        assert len(result) == 1
        assert torch.allclose(result[0], tensor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")
class TestDistributedInit:
    """分布式初始化测试"""

    def test_init_distributed_no_env(self):
        """测试没有环境变量时的初始化"""
        # 保存原始环境变量
        orig_rank = os.environ.get("RANK")
        orig_world_size = os.environ.get("WORLD_SIZE")

        try:
            # 清除环境变量
            os.environ.pop("RANK", None)
            os.environ.pop("WORLD_SIZE", None)

            # 不应该初始化
            result = init_distributed()
            assert result == False

        finally:
            # 恢复环境变量
            if orig_rank is not None:
                os.environ["RANK"] = orig_rank
            if orig_world_size is not None:
                os.environ["WORLD_SIZE"] = orig_world_size


class TestDDPEngine:
    """DDP Engine 测试（需要多卡和 torchrun）"""

    @pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason="需要至少 2 张 GPU"
    )
    def test_ddp_engine_creation(self):
        """测试 DDP Engine 创建（需要在 torchrun 下运行）"""
        if not dist.is_initialized():
            pytest.skip("需要在 torchrun 下运行此测试")

        from src.core import Engine
        from torch import nn
        import torch.optim as optim

        # 创建简单模型
        model = nn.Linear(10, 10)

        # 创建 Engine（启用 DDP）
        local_rank = get_local_rank()
        device = f"cuda:{local_rank}"

        engine = Engine(
            model=model,
            optimizer=optim.Adam(model.parameters()),
            loss_fn=nn.MSELoss(),
            device=device,
            ddp=True,
        )

        # 验证 DDP 包装
        assert engine.ddp == True
        assert hasattr(engine.model, 'module')

        # 清理
        cleanup_distributed()

    @pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason="需要至少 2 张 GPU"
    )
    def test_ddp_get_model(self):
        """测试 get_model 返回原始模型"""
        if not dist.is_initialized():
            pytest.skip("需要在 torchrun 下运行此测试")

        from src.core import Engine
        from torch import nn
        import torch.optim as optim

        model = nn.Linear(10, 10)
        local_rank = get_local_rank()
        device = f"cuda:{local_rank}"

        engine = Engine(
            model=model,
            optimizer=optim.Adam(model.parameters()),
            loss_fn=nn.MSELoss(),
            device=device,
            ddp=True,
        )

        # get_model 应该返回原始模型
        base_model = engine.get_model()
        assert isinstance(base_model, nn.Linear)
        assert not isinstance(base_model, torch.nn.parallel.DistributedDataParallel)

        cleanup_distributed()

    @pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason="需要至少 2 张 GPU"
    )
    def test_ddp_save_checkpoint(self):
        """测试 DDP 检查点保存（只在主进程保存）"""
        if not dist.is_initialized():
            pytest.skip("需要在 torchrun 下运行此测试")

        from src.core import Engine
        from torch import nn
        import torch.optim as optim

        model = nn.Linear(10, 10)
        local_rank = get_local_rank()
        device = f"cuda:{local_rank}"

        engine = Engine(
            model=model,
            optimizer=optim.Adam(model.parameters()),
            loss_fn=nn.MSELoss(),
            device=device,
            ddp=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pth"
            engine.save_checkpoint(str(path), epoch=1)

            # 只有主进程应该保存文件
            if is_main_process():
                assert path.exists()
            else:
                # 非主进程不保存
                pass

            barrier()

        cleanup_distributed()


class TestDistributedSampler:
    """分布式采样器测试"""

    def test_sampler_single_process(self):
        """单进程环境下测试采样器"""
        from src.data import get_distributed_sampler
        from torch.utils.data import TensorDataset

        if dist.is_initialized():
            pytest.skip("分布式环境已初始化，跳过单进程测试")

        # 创建简单数据集
        data = torch.randn(100, 10)
        dataset = TensorDataset(data)

        sampler = get_distributed_sampler(dataset)
        # 单进程模式应该返回 None
        assert sampler is None

    def test_create_dataloader_single_process(self):
        """单进程环境下测试 DataLoader 创建"""
        from src.data import create_dataloader
        from torch.utils.data import TensorDataset

        if dist.is_initialized():
            pytest.skip("分布式环境已初始化，跳过单进程测试")

        data = torch.randn(100, 10)
        dataset = TensorDataset(data)

        loader = create_dataloader(
            dataset,
            batch_size=10,
            shuffle=True,
            num_workers=0,
        )

        # 验证 DataLoader 正常工作
        assert len(loader) == 10

        # 验证可以迭代
        for batch in loader:
            assert batch[0].shape == (10, 10)
            break