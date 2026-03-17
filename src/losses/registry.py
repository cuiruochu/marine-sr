"""
损失函数注册机制

提供装饰器方式的损失函数注册，支持配置驱动的损失函数创建。
"""

from typing import Dict, Callable
from torch import nn


# 损失函数注册表
LOSS_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def register_loss(name: str):
    """
    注册损失函数的装饰器

    Args:
        name: 损失函数名称

    Usage:
        @register_loss("l1")
        def create_l1_loss(reduction="mean", **kwargs):
            return nn.L1Loss(reduction=reduction)
    """
    def decorator(factory: Callable[..., nn.Module]):
        LOSS_REGISTRY[name] = factory
        return factory
    return decorator


def get_loss(name: str, **kwargs) -> nn.Module:
    """
    根据名称创建损失函数

    Args:
        name: 损失函数名称
        **kwargs: 传递给损失函数的参数

    Returns:
        损失函数实例

    Raises:
        ValueError: 如果损失函数名称未注册
    """
    if name not in LOSS_REGISTRY:
        raise ValueError(
            f"未知的损失函数 '{name}'。可用选项: {list(LOSS_REGISTRY.keys())}"
        )
    return LOSS_REGISTRY[name](**kwargs)


def list_losses() -> list:
    """列出所有已注册的损失函数"""
    return list(LOSS_REGISTRY.keys())