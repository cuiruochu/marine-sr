"""
进度条回调

实现训练进度显示功能。
"""

from tqdm import tqdm
from typing import Optional
from ..core.callbacks import Callback


class ProgressCallback(Callback):
    """
    进度条回调

    功能：
    - 显示训练进度
    - 显示当前损失和学习率

    Args:
        leave: 是否保留进度条

    用法：
        callback = ProgressCallback(leave=True)
    """

    def __init__(self, leave: bool = True):
        self.leave = leave
        self._pbar: Optional[tqdm] = None

    def on_epoch_begin(self, engine, epoch: int) -> None:
        """创建进度条"""
        total_batches = len(engine._train_loader) if hasattr(engine, "_train_loader") else None
        self._pbar = tqdm(
            total=total_batches,
            desc=f"Epoch {epoch}",
            leave=self.leave,
            unit="batch",
        )

    def on_batch_end(self, engine, batch: int, logs: dict) -> None:
        """更新进度条"""
        if self._pbar is not None:
            self._pbar.update(1)
            self._pbar.set_postfix({
                "loss": f"{logs.get('loss', 0):.4f}",
                "lr": f"{engine.lr:.2e}",
            })

    def on_epoch_end(self, engine, epoch: int, logs: dict) -> None:
        """关闭进度条"""
        if self._pbar is not None:
            # 显示验证指标
            msg = (
                f"PSNR={logs.get('val_psnr', 0):.2f} "
                f"SSIM={logs.get('val_ssim', 0):.4f} "
                f"MAE={logs.get('val_mae', 0):.4f}"
            )
            self._pbar.set_postfix_str(msg)
            self._pbar.close()
            self._pbar = None