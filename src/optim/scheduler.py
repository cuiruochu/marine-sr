"""
学习率调度器工厂

提供统一的调度器创建接口，支持注册机制。

用法:
    from src.optim import get_scheduler, list_schedulers

    # 创建 StepLR 调度器
    scheduler = get_scheduler("step", optimizer, step_size=100, gamma=0.1)

    # 创建 CosineAnnealingLR 调度器
    scheduler = get_scheduler("cosine", optimizer, T_max=200)

    # 列出所有可用调度器
    print(list_schedulers())
"""

from typing import Dict, Callable, Any, Optional, List
import torch.optim as optim
from torch.optim.lr_scheduler import (
    LRScheduler,
    StepLR,
    MultiStepLR,
    CosineAnnealingLR,
    ExponentialLR,
    ReduceLROnPlateau,
    LinearLR,
    OneCycleLR,
)


# 调度器工厂函数类型
SchedulerFactory = Callable[..., LRScheduler]

# 调度器注册表
SCHEDULER_REGISTRY: Dict[str, SchedulerFactory] = {}


def register_scheduler(name: str):
    """
    注册调度器的装饰器

    Args:
        name: 调度器名称

    Usage:
        @register_scheduler("step")
        def create_step_lr(optimizer, step_size=100, gamma=0.1, **kwargs):
            return StepLR(optimizer, step_size=step_size, gamma=gamma)
    """
    def decorator(factory: SchedulerFactory):
        SCHEDULER_REGISTRY[name] = factory
        return factory
    return decorator


def get_scheduler(
    name: str,
    optimizer: optim.Optimizer,
    **kwargs
) -> Optional[LRScheduler]:
    """
    根据名称创建调度器

    Args:
        name: 调度器名称
        optimizer: 优化器
        **kwargs: 其他参数

    Returns:
        调度器实例，如果 name 为空或 "none" 则返回 None

    Raises:
        ValueError: 如果调度器名称未注册
    """
    if not name or name.lower() == "none":
        return None

    if name not in SCHEDULER_REGISTRY:
        raise ValueError(
            f"未知的调度器 '{name}'。可用选项: {list(SCHEDULER_REGISTRY.keys())}"
        )
    return SCHEDULER_REGISTRY[name](optimizer, **kwargs)


def list_schedulers() -> list:
    """列出所有已注册的调度器"""
    return list(SCHEDULER_REGISTRY.keys())


# ============ 内置调度器 ============

@register_scheduler("step")
def create_step_lr(
    optimizer: optim.Optimizer,
    step_size: int = 100,
    gamma: float = 0.1,
    **kwargs
) -> StepLR:
    """
    创建 StepLR 调度器

    Args:
        optimizer: 优化器
        step_size: 每 N 个 epoch 降低学习率
        gamma: 学习率衰减因子
    """
    return StepLR(optimizer, step_size=step_size, gamma=gamma)


@register_scheduler("multistep")
def create_multistep_lr(
    optimizer: optim.Optimizer,
    milestones: List[int] = None,
    gamma: float = 0.1,
    **kwargs
) -> MultiStepLR:
    """
    创建 MultiStepLR 调度器

    Args:
        optimizer: 优化器
        milestones: 降低学习率的 epoch 列表
        gamma: 学习率衰减因子
    """
    milestones = milestones or [100, 200]
    return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)


@register_scheduler("cosine")
def create_cosine_lr(
    optimizer: optim.Optimizer,
    T_max: int = 200,
    eta_min: float = 0,
    **kwargs
) -> CosineAnnealingLR:
    """
    创建 CosineAnnealingLR 调度器

    Args:
        optimizer: 优化器
        T_max: 周期长度
        eta_min: 最小学习率
    """
    return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)


@register_scheduler("exponential")
def create_exponential_lr(
    optimizer: optim.Optimizer,
    gamma: float = 0.95,
    **kwargs
) -> ExponentialLR:
    """
    创建 ExponentialLR 调度器

    Args:
        optimizer: 优化器
        gamma: 学习率衰减因子
    """
    return ExponentialLR(optimizer, gamma=gamma)


@register_scheduler("plateau")
def create_reduce_on_plateau(
    optimizer: optim.Optimizer,
    mode: str = "min",
    factor: float = 0.1,
    patience: int = 10,
    min_lr: float = 1e-7,
    **kwargs
) -> ReduceLROnPlateau:
    """
    创建 ReduceLROnPlateau 调度器

    Args:
        optimizer: 优化器
        mode: "min" 或 "max"
        factor: 学习率衰减因子
        patience: 等待 N 个 epoch 无改善后降低学习率
        min_lr: 最小学习率
    """
    return ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=factor,
        patience=patience,
        min_lr=min_lr,
        **kwargs
    )


@register_scheduler("linear")
def create_linear_lr(
    optimizer: optim.Optimizer,
    start_factor: float = 1.0,
    end_factor: float = 0.0,
    total_iters: int = 100,
    **kwargs
) -> LinearLR:
    """
    创建 LinearLR 调度器

    Args:
        optimizer: 优化器
        start_factor: 起始因子
        end_factor: 结束因子
        total_iters: 总迭代次数
    """
    return LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=end_factor,
        total_iters=total_iters,
    )


@register_scheduler("onecycle")
def create_onecycle_lr(
    optimizer: optim.Optimizer,
    max_lr: float = 1e-3,
    total_steps: int = 1000,
    pct_start: float = 0.3,
    **kwargs
) -> OneCycleLR:
    """
    创建 OneCycleLR 调度器

    Args:
        optimizer: 优化器
        max_lr: 最大学习率
        total_steps: 总步数
        pct_start: 上升阶段占比
    """
    return OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        **kwargs
    )