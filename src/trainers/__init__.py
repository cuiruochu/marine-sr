"""
训练器模块

提供统一的训练接口。
"""

from .common_trainer import Trainer, CommonTrainer, create_trainer, get_trainer

__all__ = [
    "Trainer",
    "CommonTrainer",
    "create_trainer",
    "get_trainer",
]