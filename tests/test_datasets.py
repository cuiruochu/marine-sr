"""
数据集单元测试

测试数据加载、归一化、MWD 编码等。
"""

import pytest
import torch
import numpy as np
import tempfile
import os

from src.datasets.dataset import (
    MarineTrainSet,
    MarineTestSet,
    _is_mwd_param,
)
from src.datasets.utils import encode_mwd, normalize


class TestHelperFunctions:
    """测试辅助函数"""

    def test_is_mwd_param(self):
        """测试 _is_mwd_param 函数"""
        assert _is_mwd_param("mwd") is True
        assert _is_mwd_param("MWD") is True
        assert _is_mwd_param("Mwd") is True
        assert _is_mwd_param("wind") is False
        assert _is_mwd_param("mwp") is False
        assert _is_mwd_param("swh") is False


class TestEncodeMWD:
    """测试 MWD 编码"""

    def test_encode_mwd_shape(self):
        """测试 MWD 编码输出 shape"""
        # 单通道输入 (1, H, W)
        hr = torch.randn(1, 64, 64)
        encoded = encode_mwd(hr)

        # 输出应为 2 通道
        assert encoded.shape == (2, 64, 64)

    def test_encode_mwd_zero_degree(self):
        """测试 MWD 编码 0 度"""
        # 输入值直接表示角度
        hr = torch.zeros(1, 10, 10)  # 0 度
        encoded = encode_mwd(hr)

        cos_channel = encoded[0]
        sin_channel = encoded[1]

        # cos(0°) = 1, sin(0°) = 0
        assert torch.allclose(cos_channel, torch.ones_like(cos_channel), atol=1e-6)
        assert torch.allclose(sin_channel, torch.zeros_like(sin_channel), atol=1e-6)

    def test_encode_mwd_90_degree(self):
        """测试 MWD 编码 90 度"""
        # 输入值直接表示角度
        hr = torch.ones(1, 10, 10) * 90.0  # 90 度
        encoded = encode_mwd(hr)

        cos_channel = encoded[0]
        sin_channel = encoded[1]

        # cos(90°) ≈ 0, sin(90°) = 1
        assert torch.allclose(cos_channel, torch.zeros_like(cos_channel), atol=1e-6)
        assert torch.allclose(sin_channel, torch.ones_like(sin_channel), atol=1e-6)


class TestNormalize:
    """测试归一化"""

    def test_normalize_shape(self):
        """测试归一化不改变 shape"""
        mean = torch.tensor([1.0]).unsqueeze(-1).unsqueeze(-1)
        std = torch.tensor([2.0]).unsqueeze(-1).unsqueeze(-1)

        x = torch.randn(1, 64, 64)
        normalized = normalize(mean, std, x)

        assert normalized.shape == x.shape

    def test_normalize_values(self):
        """测试归一化值正确"""
        mean = torch.tensor([10.0]).unsqueeze(-1).unsqueeze(-1)
        std = torch.tensor([2.0]).unsqueeze(-1).unsqueeze(-1)

        x = torch.ones(1, 4, 4) * 12.0  # (12 - 10) / 2 = 1
        normalized = normalize(mean, std, x)

        expected = torch.ones(1, 4, 4)
        assert torch.allclose(normalized, expected)

    def test_normalize_zero_std(self):
        """测试 std=1 时归一化"""
        mean = torch.tensor([5.0]).unsqueeze(-1).unsqueeze(-1)
        std = torch.tensor([1.0]).unsqueeze(-1).unsqueeze(-1)

        x = torch.ones(1, 4, 4) * 8.0
        normalized = normalize(mean, std, x)

        expected = torch.ones(1, 4, 4) * 3.0
        assert torch.allclose(normalized, expected)


class TestMarineTestSet:
    """测试 MarineTestSet"""

    @pytest.fixture
    def temp_data_dir(self):
        """创建临时数据目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建测试数据文件
            for i in range(3):
                data = np.random.rand(64, 64).astype(np.float32)
                np.save(os.path.join(tmpdir, f"test_{i}.npy"), data)
            yield tmpdir

    def test_testset_length(self, temp_data_dir):
        """测试数据集长度"""
        testset = MarineTestSet(
            hr_root=temp_data_dir,
            upscale=2,
            mean=[0.5],
            std=[1.0],
            is_mwd=False,
            sample_q=False
        )

        assert len(testset) == 3

    def test_testset_getitem_shape(self, temp_data_dir):
        """测试 __getitem__ 返回 shape"""
        testset = MarineTestSet(
            hr_root=temp_data_dir,
            upscale=2,
            mean=[0.5],
            std=[1.0],
            is_mwd=False,
            sample_q=False
        )

        lr, hr, filename = testset[0]

        # HR 保持原始大小
        assert hr.shape == (1, 64, 64)
        # LR 是 HR 的 1/upscale
        assert lr.shape == (1, 32, 32)

    def test_testset_sample_q(self, temp_data_dir):
        """测试 sample_q 限制样本数"""
        testset = MarineTestSet(
            hr_root=temp_data_dir,
            upscale=2,
            mean=[0.5],
            std=[1.0],
            is_mwd=False,
            sample_q=2
        )

        assert len(testset) == 2

    def test_testset_mwd_encoding(self, temp_data_dir):
        """测试 MWD 编码在数据集中"""
        testset = MarineTestSet(
            hr_root=temp_data_dir,
            upscale=2,
            mean=[0.5, 0.5],
            std=[1.0, 1.0],
            is_mwd=True,
            sample_q=False
        )

        lr, hr, filename = testset[0]

        # MWD 编码后应为 2 通道
        assert hr.shape[0] == 2
        assert lr.shape[0] == 2


class TestMarineTrainSet:
    """测试 MarineTrainSet"""

    @pytest.fixture
    def temp_data_dir(self):
        """创建临时数据目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建足够大的测试数据（需要能裁剪 patch）
            for i in range(5):
                data = np.random.rand(120, 120).astype(np.float32)
                np.save(os.path.join(tmpdir, f"train_{i}.npy"), data)
            yield tmpdir

    def test_trainset_length(self, temp_data_dir):
        """测试训练数据集长度"""
        trainset = MarineTrainSet(
            hr_root=temp_data_dir,
            upscale=2,
            lr_patch_size=30,
            mean=[0.5],
            std=[1.0],
            is_mwd=False
        )

        assert len(trainset) == 5

    def test_trainset_getitem_shape(self, temp_data_dir):
        """测试训练数据集 __getitem__ 返回 shape"""
        trainset = MarineTrainSet(
            hr_root=temp_data_dir,
            upscale=2,
            lr_patch_size=30,
            mean=[0.5],
            std=[1.0],
            is_mwd=False
        )

        lr, hr = trainset[0]

        # HR patch: lr_patch_size * upscale
        assert hr.shape == (1, 60, 60)
        # LR patch
        assert lr.shape == (1, 30, 30)

    def test_trainset_mwd(self, temp_data_dir):
        """测试训练数据集 MWD 模式"""
        trainset = MarineTrainSet(
            hr_root=temp_data_dir,
            upscale=2,
            lr_patch_size=30,
            mean=[0.5, 0.5],
            std=[1.0, 1.0],
            is_mwd=True
        )

        lr, hr = trainset[0]

        # MWD 应为 2 通道
        assert hr.shape[0] == 2
        assert lr.shape[0] == 2