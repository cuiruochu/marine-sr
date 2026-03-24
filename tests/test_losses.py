"""
损失函数模块测试
"""

import pytest
import torch
import torch.nn as nn

from src.losses import (
    LOSS_REGISTRY,
    get_loss,
    list_losses,
)


class TestLossRegistry:
    """损失函数注册测试"""

    def test_list_losses(self):
        """测试列出损失函数"""
        losses = list_losses()
        assert "l1" in losses
        assert "l2" in losses
        assert "mse" in losses
        assert "mae" in losses

    def test_registry_contains_expected_losses(self):
        """测试注册表包含预期的损失函数"""
        assert "l1" in LOSS_REGISTRY
        assert "l2" in LOSS_REGISTRY
        assert "mse" in LOSS_REGISTRY
        assert "mae" in LOSS_REGISTRY


class TestGetLoss:
    """get_loss 函数测试"""

    def test_get_l1_loss(self):
        """测试创建 L1 Loss"""
        loss_fn = get_loss("l1")
        assert isinstance(loss_fn, nn.L1Loss)

    def test_get_l2_loss(self):
        """测试创建 L2 Loss"""
        loss_fn = get_loss("l2")
        assert isinstance(loss_fn, nn.MSELoss)

    def test_get_mse_loss(self):
        """测试创建 MSE Loss（L2 别名）"""
        loss_fn = get_loss("mse")
        assert isinstance(loss_fn, nn.MSELoss)

    def test_get_mae_loss(self):
        """测试创建 MAE Loss（L1 别名）"""
        loss_fn = get_loss("mae")
        assert isinstance(loss_fn, nn.L1Loss)

    def test_get_loss_with_reduction(self):
        """测试带 reduction 参数"""
        loss_fn = get_loss("l1", reduction="sum")
        assert isinstance(loss_fn, nn.L1Loss)
        # 验证 reduction 参数生效
        pred = torch.randn(2, 3)
        target = torch.randn(2, 3)
        loss = loss_fn(pred, target)
        assert loss.ndim == 0  # 标量

    def test_get_loss_unknown_name(self):
        """测试未知损失函数名称"""
        with pytest.raises(ValueError) as excinfo:
            get_loss("unknown_loss")
        assert "未知的损失函数" in str(excinfo.value)


class TestLossForward:
    """损失函数前向传播测试"""

    def test_l1_forward(self):
        """测试 L1 Loss 前向传播"""
        loss_fn = get_loss("l1")
        pred = torch.randn(4, 3, 32, 32)
        target = torch.randn(4, 3, 32, 32)
        loss = loss_fn(pred, target)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_l2_forward(self):
        """测试 L2 Loss 前向传播"""
        loss_fn = get_loss("l2")
        pred = torch.randn(4, 3, 32, 32)
        target = torch.randn(4, 3, 32, 32)
        loss = loss_fn(pred, target)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_l1_zero_loss(self):
        """测试 L1 Loss 零损失"""
        loss_fn = get_loss("l1")
        pred = torch.ones(2, 3)
        target = torch.ones(2, 3)
        loss = loss_fn(pred, target)
        assert loss.item() == 0.0

    def test_l2_zero_loss(self):
        """测试 L2 Loss 零损失"""
        loss_fn = get_loss("l2")
        pred = torch.ones(2, 3)
        target = torch.ones(2, 3)
        loss = loss_fn(pred, target)
        assert loss.item() == 0.0
