"""
检查点回调

实现模型保存功能。
"""

import shutil
from pathlib import Path
from typing import Optional, Literal
from ..core.callbacks import Callback


class CheckpointCallback(Callback):
    """
    模型保存回调

    功能：
    - 定期保存检查点
    - 保存最佳模型
    - 保留最近 N 个检查点

    Args:
        save_dir: 保存目录
        every: 每 N 个 epoch 保存（0 表示不定期保存）
        save_best: 是否保存最佳模型
        monitor: 监控指标
        mode: "min" 或 "max"，指标优化方向
        keep_last: 保留最近 N 个检查点（0 表示全部保留）

    用法：
        callback = CheckpointCallback(
            save_dir="checkpoints/edsr_wind_x2",
            every=10,
            save_best=True,
            monitor="val_mae",
            mode="min"
        )
    """

    def __init__(
        self,
        save_dir: str,
        every: int = 10,
        save_best: bool = True,
        monitor: str = "val_mae",
        mode: Literal["min", "max"] = "min",
        keep_last: int = 0,
    ):
        self.save_dir = Path(save_dir)
        self.every = every
        self.save_best = save_best
        self.monitor = monitor
        self.mode = mode
        self.keep_last = keep_last

        # 初始化最佳值
        if mode == "min":
            self.best_value = float("inf")
            self.is_better = lambda current, best: current < best
        else:
            self.best_value = float("-inf")
            self.is_better = lambda current, best: current > best

        # 保存历史
        self._saved_epochs: list = []

    def on_train_begin(self, engine) -> None:
        """创建保存目录"""
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, engine, epoch: int, logs: dict) -> None:
        """保存检查点"""
        # 定期保存
        if self.every > 0 and epoch % self.every == 0:
            path = self.save_dir / f"epoch_{epoch}.pth"
            engine.save_checkpoint(str(path), epoch)
            self._saved_epochs.append(epoch)

            # 清理旧检查点
            if self.keep_last > 0 and len(self._saved_epochs) > self.keep_last:
                old_epoch = self._saved_epochs.pop(0)
                old_path = self.save_dir / f"epoch_{old_epoch}.pth"
                if old_path.exists():
                    old_path.unlink()

        # 保存最佳模型
        if self.save_best and self.monitor in logs:
            current = logs[self.monitor]
            if self.is_better(current, self.best_value):
                self.best_value = current
                engine.best_metric = current
                engine.best_epoch = epoch

                # 保存最佳模型
                path = self.save_dir / "best.pth"
                engine.save_checkpoint(str(path), epoch)

    def on_train_end(self, engine) -> None:
        """训练结束时保存最终模型"""
        path = self.save_dir / "last.pth"
        engine.save_checkpoint(str(path), engine.current_epoch)