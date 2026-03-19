"""
优化器和调度器模块测试
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from src.optim import (
    get_optimizer,
    get_scheduler,
    list_optimizers,
    list_schedulers,
    OPTIMIZER_REGISTRY,
    SCHEDULER_REGISTRY,
)


class DummyModel(nn.Module):
    """测试用简单模型"""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


class TestOptimizerRegistry:
    """优化器注册测试"""

    def test_list_optimizers(self):
        """测试列出优化器"""
        optimizers = list_optimizers()
        assert "adam" in optimizers
        assert "adamw" in optimizers
        assert "sgd" in optimizers
        assert "rmsprop" in optimizers

    def test_registry_contains_expected_optimizers(self):
        """测试注册表包含预期的优化器"""
        assert "adam" in OPTIMIZER_REGISTRY
        assert "adamw" in OPTIMIZER_REGISTRY
        assert "sgd" in OPTIMIZER_REGISTRY


class TestGetOptimizer:
    """get_optimizer 函数测试"""

    def test_get_adam(self):
        """测试创建 Adam 优化器"""
        model = DummyModel()
        optimizer = get_optimizer("adam", model, lr=1e-3)
        assert isinstance(optimizer, optim.Adam)
        assert optimizer.defaults["lr"] == 1e-3

    def test_get_adamw(self):
        """测试创建 AdamW 优化器"""
        model = DummyModel()
        optimizer = get_optimizer("adamw", model, lr=1e-4, weight_decay=0.01)
        assert isinstance(optimizer, optim.AdamW)
        assert optimizer.defaults["weight_decay"] == 0.01

    def test_get_sgd(self):
        """测试创建 SGD 优化器"""
        model = DummyModel()
        optimizer = get_optimizer("sgd", model, lr=1e-2, momentum=0.9)
        assert isinstance(optimizer, optim.SGD)
        assert optimizer.defaults["momentum"] == 0.9

    def test_get_rmsprop(self):
        """测试创建 RMSprop 优化器"""
        model = DummyModel()
        optimizer = get_optimizer("rmsprop", model, lr=1e-3)
        assert isinstance(optimizer, optim.RMSprop)

    def test_get_optimizer_unknown_name(self):
        """测试未知优化器名称"""
        model = DummyModel()
        with pytest.raises(ValueError) as excinfo:
            get_optimizer("unknown_optimizer", model)
        assert "未知的优化器" in str(excinfo.value)


class TestSchedulerRegistry:
    """调度器注册测试"""

    def test_list_schedulers(self):
        """测试列出调度器"""
        schedulers = list_schedulers()
        assert "step" in schedulers
        assert "cosine" in schedulers
        assert "multistep" in schedulers
        assert "plateau" in schedulers

    def test_registry_contains_expected_schedulers(self):
        """测试注册表包含预期的调度器"""
        assert "step" in SCHEDULER_REGISTRY
        assert "cosine" in SCHEDULER_REGISTRY
        assert "multistep" in SCHEDULER_REGISTRY
        assert "exponential" in SCHEDULER_REGISTRY


class TestGetScheduler:
    """get_scheduler 函数测试"""

    def test_get_step_lr(self):
        """测试创建 StepLR 调度器"""
        model = DummyModel()
        optimizer = get_optimizer("adam", model)
        scheduler = get_scheduler("step", optimizer, step_size=10, gamma=0.5)
        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)

    def test_get_cosine_lr(self):
        """测试创建 CosineAnnealingLR 调度器"""
        model = DummyModel()
        optimizer = get_optimizer("adam", model)
        scheduler = get_scheduler("cosine", optimizer, T_max=100, eta_min=1e-6)
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_get_multistep_lr(self):
        """测试创建 MultiStepLR 调度器"""
        model = DummyModel()
        optimizer = get_optimizer("adam", model)
        scheduler = get_scheduler("multistep", optimizer, milestones=[50, 100], gamma=0.1)
        assert isinstance(scheduler, torch.optim.lr_scheduler.MultiStepLR)

    def test_get_plateau(self):
        """测试创建 ReduceLROnPlateau 调度器"""
        model = DummyModel()
        optimizer = get_optimizer("adam", model)
        scheduler = get_scheduler("plateau", optimizer, mode="min", patience=10)
        assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_get_scheduler_none(self):
        """测试空调度器"""
        model = DummyModel()
        optimizer = get_optimizer("adam", model)
        scheduler = get_scheduler("none", optimizer)
        assert scheduler is None

        scheduler = get_scheduler("", optimizer)
        assert scheduler is None

    def test_get_scheduler_unknown_name(self):
        """测试未知调度器名称"""
        model = DummyModel()
        optimizer = get_optimizer("adam", model)
        with pytest.raises(ValueError) as excinfo:
            get_scheduler("unknown_scheduler", optimizer)
        assert "未知的调度器" in str(excinfo.value)


class TestSchedulerStep:
    """调度器步进测试"""

    def test_step_lr_step(self):
        """测试 StepLR 步进"""
        model = DummyModel()
        optimizer = get_optimizer("adam", model, lr=1e-3)
        scheduler = get_scheduler("step", optimizer, step_size=2, gamma=0.1)

        initial_lr = optimizer.param_groups[0]["lr"]
        assert initial_lr == 1e-3

        # 第一步
        optimizer.step()
        scheduler.step()
        assert optimizer.param_groups[0]["lr"] == initial_lr  # 未到 step_size

        # 第二步
        optimizer.step()
        scheduler.step()
        assert optimizer.param_groups[0]["lr"] == initial_lr * 0.1  # 到达 step_size

    def test_cosine_lr_step(self):
        """测试 CosineAnnealingLR 步进"""
        model = DummyModel()
        optimizer = get_optimizer("adam", model, lr=1e-3)
        scheduler = get_scheduler("cosine", optimizer, T_max=10, eta_min=1e-5)

        initial_lr = optimizer.param_groups[0]["lr"]

        # 步进后学习率应下降
        optimizer.step()
        scheduler.step()
        assert optimizer.param_groups[0]["lr"] < initial_lr
