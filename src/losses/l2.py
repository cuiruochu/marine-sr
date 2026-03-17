"""
L2 Loss (Mean Squared Error)
"""

import torch.nn as nn
from .registry import register_loss


@register_loss("l2")
def create_l2_loss(reduction: str = "mean", **kwargs) -> nn.MSELoss:
    """
    创建 L2 Loss

    Args:
        reduction: 归约方式 ("mean", "sum", "none")

    Returns:
        MSELoss 实例
    """
    return nn.MSELoss(reduction=reduction)


@register_loss("mse")
def create_mse_loss(reduction: str = "mean", **kwargs) -> nn.MSELoss:
    """
    创建 MSE Loss (L2 的别名)

    Args:
        reduction: 归约方式

    Returns:
        MSELoss 实例
    """
    return nn.MSELoss(reduction=reduction)


@register_loss("mse")
def create_mse_loss(reduction: str = "mean", **kwargs) -> nn.MSELoss:
    """MSE 是 L2 的别名"""
    return nn.MSELoss(reduction=reduction)