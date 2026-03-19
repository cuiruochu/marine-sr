"""
评估指标回调

在评估过程中累计计算 PSNR、SSIM、MAE 指标，并在评估结束时记录日志。
"""

from typing import Dict, Any, Optional
import torch
import numpy as np

from src.core.callbacks import Callback
from src.utils import get_logger


class MetricsCallback(Callback):
    """
    指标回调

    在评估过程中累计计算 PSNR、SSIM、MAE 指标，
    并在评估结束时记录日志。

    用法：
        evaluator = Evaluator(
            model=model,
            callbacks=[MetricsCallback()]
        )
    """

    def __init__(self, verbose: bool = True):
        """
        初始化

        Args:
            verbose: 是否输出指标日志
        """
        self.verbose = verbose
        self.logger = get_logger()

    def on_eval_begin(self, evaluator: Any) -> None:
        """评估开始时重置累计器"""
        self.total_psnr = 0.0
        self.total_ssim = 0.0
        self.total_mae = 0.0
        self.max_mae = 0.0
        self.num_batches = 0

    def on_batch_end(
        self,
        evaluator: Any,
        batch_idx: int,
        logs: Optional[Dict[str, Any]] = None,
        sr: torch.Tensor = None,
        hr: torch.Tensor = None,
        filename: str = None,
        metrics: Dict[str, float] = None,
        **kwargs,
    ) -> None:
        """累计指标"""
        if metrics is None:
            return

        self.total_psnr += metrics.get("psnr", 0.0)
        self.total_ssim += metrics.get("ssim", 0.0)
        self.total_mae += metrics.get("mae", 0.0)
        self.max_mae = max(self.max_mae, metrics.get("max_mae", 0.0))
        self.num_batches += 1

    def on_eval_end(self, evaluator: Any, metrics: Dict[str, float]) -> None:
        """评估结束时输出指标日志"""
        if not self.verbose or not metrics:
            return

        self.logger.info("=" * 50)
        self.logger.info("Evaluation Results")
        self.logger.info("=" * 50)
        self.logger.info(f"  PSNR:    {metrics['psnr']:.4f} dB")
        self.logger.info(f"  SSIM:    {metrics['ssim']:.4f}")
        self.logger.info(f"  MAE:     {metrics['mae']:.4f}")
        self.logger.info(f"  Max MAE: {metrics['max_mae']:.4f}")
        self.logger.info("=" * 50)
