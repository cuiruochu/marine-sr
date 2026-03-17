"""
学习率监控回调

实现学习率变化监控和记录功能。
"""

from typing import List, Dict
from ..core.callbacks import Callback


class LRMonitorCallback(Callback):
    """
    学习率监控回调

    功能：
    - 记录学习率变化
    - 检测学习率异常

    Args:
        log_freq: 记录频率（每 N 个 batch）

    用法：
        callback = LRMonitorCallback(log_freq=100)
    """

    def __init__(self, log_freq: int = 100):
        self.log_freq = log_freq
        self.lr_history: List[Dict[str, float]] = []

    def on_batch_end(self, engine, batch: int, logs: dict) -> None:
        """记录学习率"""
        if batch % self.log_freq != 0:
            return

        # 记录所有参数组的学习率
        for i, pg in enumerate(engine.optimizer.param_groups):
            self.lr_history.append({
                "step": engine.global_step,
                "epoch": engine.current_epoch,
                "param_group": i,
                "lr": pg["lr"],
            })

    def on_epoch_end(self, engine, epoch: int, logs: dict) -> None:
        """在 epoch 结束时检查学习率"""
        current_lr = engine.lr
        # 添加到 epoch logs
        logs["lr"] = current_lr

    def get_lr_schedule(self) -> List[Dict[str, float]]:
        """获取学习率调度历史"""
        return self.lr_history.copy()