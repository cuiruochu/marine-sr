"""
优化器工厂

提供统一的优化器创建接口，支持注册机制。

用法:
    from src.optim import get_optimizer, list_optimizers

    # 创建 Adam 优化器
    optimizer = get_optimizer("adam", model, lr=1e-4)

    # 创建带 weight_decay 的 AdamW
    optimizer = get_optimizer("adamw", model, lr=1e-4, weight_decay=0.01)

    # 列出所有可用优化器
    logger.info(list_optimizers())
"""

from typing import Dict, Callable, Any
import torch.optim as optim
from torch.nn import Module


# 优化器工厂函数类型
OptimizerFactory = Callable[..., optim.Optimizer]

# 优化器注册表
OPTIMIZER_REGISTRY: Dict[str, OptimizerFactory] = {}


def register_optimizer(name: str):
    """
    注册优化器的装饰器

    Args:
        name: 优化器名称

    Usage:
        @register_optimizer("adam")
        def create_adam(model, lr=1e-3, **kwargs):
            return optim.Adam(model.parameters(), lr=lr, **kwargs)
    """
    def decorator(factory: OptimizerFactory):
        OPTIMIZER_REGISTRY[name] = factory
        return factory
    return decorator


def get_optimizer(
    name: str,
    model: Module,
    lr: float = 1e-3,
    **kwargs
) -> optim.Optimizer:
    """
    根据名称创建优化器

    Args:
        name: 优化器名称
        model: 模型
        lr: 学习率
        **kwargs: 其他参数（如 weight_decay, betas 等）

    Returns:
        优化器实例

    Raises:
        ValueError: 如果优化器名称未注册
    """
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(
            f"未知的优化器 '{name}'。可用选项: {list(OPTIMIZER_REGISTRY.keys())}"
        )
    return OPTIMIZER_REGISTRY[name](model, lr=lr, **kwargs)


def list_optimizers() -> list:
    """列出所有已注册的优化器"""
    return list(OPTIMIZER_REGISTRY.keys())


# ============ 内置优化器 ============

@register_optimizer("adam")
def create_adam(model: Module, lr: float = 1e-3, **kwargs) -> optim.Adam:
    """
    创建 Adam 优化器

    Args:
        model: 模型
        lr: 学习率
        **kwargs: 其他参数（betas, eps, weight_decay 等）
    """
    return optim.Adam(model.parameters(), lr=lr, **kwargs)


@register_optimizer("adamw")
def create_adamw(model: Module, lr: float = 1e-3, **kwargs) -> optim.AdamW:
    """
    创建 AdamW 优化器

    Args:
        model: 模型
        lr: 学习率
        **kwargs: 其他参数
    """
    return optim.AdamW(model.parameters(), lr=lr, **kwargs)


@register_optimizer("sgd")
def create_sgd(
    model: Module,
    lr: float = 1e-3,
    momentum: float = 0.9,
    **kwargs
) -> optim.SGD:
    """
    创建 SGD 优化器

    Args:
        model: 模型
        lr: 学习率
        momentum: 动量
        **kwargs: 其他参数
    """
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum, **kwargs)


@register_optimizer("rmsprop")
def create_rmsprop(model: Module, lr: float = 1e-3, **kwargs) -> optim.RMSprop:
    """
    创建 RMSprop 优化器

    Args:
        model: 模型
        lr: 学习率
        **kwargs: 其他参数
    """
    return optim.RMSprop(model.parameters(), lr=lr, **kwargs)


@register_optimizer("adagrad")
def create_adagrad(model: Module, lr: float = 1e-2, **kwargs) -> optim.Adagrad:
    """
    创建 Adagrad 优化器

    Args:
        model: 模型
        lr: 学习率
        **kwargs: 其他参数
    """
    return optim.Adagrad(model.parameters(), lr=lr, **kwargs)
