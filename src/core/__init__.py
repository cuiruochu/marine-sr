"""
核心模块

提供训练引擎、评估引擎、回调机制和指标计算。
"""

from .callbacks import Callback, CallbackList
from .engine import Engine
from .evaluator import Evaluator
from .model_output import ModelOutput, normalize_model_output, sum_aux_losses
from .metrics import (
    calculate_psnr,
    calculate_ssim,
    calculate_mae,
    calculate_max_mae,
    reverse_norm,
    normalize_to_01,
    apply_mask,
)

__all__ = [
    "Callback",
    "CallbackList",
    "Engine",
    "Evaluator",
    "ModelOutput",
    "normalize_model_output",
    "sum_aux_losses",
    "calculate_psnr",
    "calculate_ssim",
    "calculate_mae",
    "calculate_max_mae",
    "reverse_norm",
    "normalize_to_01",
    "apply_mask",
]
