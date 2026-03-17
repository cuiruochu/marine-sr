"""
优化器模块

提供可配置的优化器和调度器创建，支持注册机制。

用法:
    from src.optim import get_optimizer, get_scheduler, list_optimizers, list_schedulers

    # 创建优化器
    optimizer = get_optimizer("adam", model, lr=1e-4)
    optimizer = get_optimizer("adamw", model, lr=1e-4, weight_decay=0.01)

    # 创建调度器
    scheduler = get_scheduler("step", optimizer, step_size=100, gamma=0.1)
    scheduler = get_scheduler("cosine", optimizer, T_max=200)

    # 列出可用选项
    print(list_optimizers())
    print(list_schedulers())
"""

from .optimizer import (
    OPTIMIZER_REGISTRY,
    register_optimizer,
    get_optimizer,
    list_optimizers,
)
from .scheduler import (
    SCHEDULER_REGISTRY,
    register_scheduler,
    get_scheduler,
    list_schedulers,
)

__all__ = [
    # 优化器
    "OPTIMIZER_REGISTRY",
    "register_optimizer",
    "get_optimizer",
    "list_optimizers",
    # 调度器
    "SCHEDULER_REGISTRY",
    "register_scheduler",
    "get_scheduler",
    "list_schedulers",
]