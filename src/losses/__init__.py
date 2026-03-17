"""
损失函数模块

提供可配置的损失函数创建，支持注册机制。

用法:
    from src.losses import get_loss, list_losses

    # 创建 L1 Loss
    loss_fn = get_loss("l1")

    # 创建带参数的 L2 Loss
    loss_fn = get_loss("l2", reduction="sum")

    # 列出所有可用损失函数
    print(list_losses())
"""

from .registry import LOSS_REGISTRY, register_loss, get_loss, list_losses

# 导入具体实现以触发注册
from .l1 import create_l1_loss, create_mae_loss
from .l2 import create_l2_loss, create_mse_loss

__all__ = [
    "LOSS_REGISTRY",
    "register_loss",
    "get_loss",
    "list_losses",
    "create_l1_loss",
    "create_l2_loss",
    "create_mae_loss",
    "create_mse_loss",
]