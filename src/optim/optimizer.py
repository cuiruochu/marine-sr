"""
优化器模块
"""

from torch import optim

OPTIMIZER_REGISTRY = {
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "sgd": optim.SGD,
}


def register_optimizer(name: str):
    """注册优化器的装饰器。"""

    def decorator(cls):
        OPTIMIZER_REGISTRY[name] = cls
        return cls

    return decorator


def get_optimizer(name: str, model, lr: float, **kwargs):
    """创建优化器实例。"""
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(f"优化器 '{name}' 未注册。可用: {list(OPTIMIZER_REGISTRY.keys())}")
    return OPTIMIZER_REGISTRY[name](model.parameters(), lr=lr, **kwargs)


def list_optimizers() -> list:
    return list(OPTIMIZER_REGISTRY.keys())
