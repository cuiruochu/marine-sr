"""
日志回调

实现训练日志记录功能。
"""

from typing import Optional
from ..core.callbacks import Callback


class LoggingCallback(Callback):
    """
    日志记录回调

    功能：
    - 记录每个 epoch 的指标
    - 支持文件输出

    Args:
        log_file: 日志文件路径（可选）
        log_freq: 日志频率（每 N 个 epoch）

    用法：
        callback = LoggingCallback(
            log_file="experiments/train.log",
            log_freq=1
        )
    """

    def __init__(
        self,
        log_file: Optional[str] = None,
        log_freq: int = 1,
    ):
        self.log_file = log_file
        self.log_freq = log_freq

        # 使用现有日志系统
        try:
            from src.utils import get_logger
            self.logger = get_logger()
        except ImportError:
            import logging
            self.logger = logging.getLogger("marine_sr")

    def on_epoch_end(self, engine, epoch: int, logs: dict) -> None:
        """记录 epoch 指标"""
        if epoch % self.log_freq != 0:
            return

        # 格式化指标
        metrics = []
        for k, v in logs.items():
            if isinstance(v, float):
                metrics.append(f"{k}={v:.4f}")
            else:
                metrics.append(f"{k}={v}")

        msg = f"[Epoch {epoch}] " + " | ".join(metrics)
        self.logger.info(msg)

    def on_train_begin(self, engine) -> None:
        """记录训练开始"""
        total_params = engine.count_parameters()
        self.logger.info(f"Training started")
        self.logger.info(f"Model parameters: {total_params:,}")
        self.logger.info(f"Device: {engine.device}")

    def on_train_end(self, engine) -> None:
        """记录训练结束"""
        self.logger.info(
            f"Training completed. "
            f"Best {getattr(engine, '_monitor', 'val_mae')}: {engine.best_metric:.4f} "
            f"at epoch {engine.best_epoch}"
        )