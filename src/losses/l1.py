"""
L1 Loss (Mean Absolute Error)
"""

import torch.nn as nn
from .registry import register_loss


@register_loss("l1")
def create_l1_loss(reduction: str = "mean", **kwargs) -> nn.L1Loss:
    """
    创建 L1 Loss

    Args:
        reduction: 归约方式 ("mean", "sum", "none")

    Returns:
        L1Loss 实例
    """
    return nn.L1Loss(reduction=reduction)


@register_loss("mae")
def create_mae_loss(reduction: str = "mean", **kwargs) -> nn.L1Loss:
    """
    创建 MAE Loss (L1 的别名)

    Args:
        reduction: 归约方式

    Returns:
        L1Loss 实例
    """
    return nn.L1Loss(reduction=reduction)