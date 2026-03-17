"""
WandB 回调

实现 Weights & Biases 日志记录功能。
"""

from typing import Optional, Dict, Any
from ..core.callbacks import Callback

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class WandbCallback(Callback):
    """
    WandB 日志回调

    功能：
    - 记录训练损失
    - 记录验证指标
    - 记录学习率

    Args:
        project: WandB 项目名
        name: 实验名称
        config: 配置字典
        mode: "online", "offline", "disabled"
        log_freq: batch 日志频率

    用法：
        callback = WandbCallback(
            project="marine-sr",
            name="edsr_wind_x2",
            config={"model": "EDSR", "epochs": 200},
            mode="offline"
        )
    """

    def __init__(
        self,
        project: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        mode: str = "offline",
        log_freq: int = 100,
    ):
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is not installed. Run: pip install wandb")

        self.project = project
        self.name = name
        self.config = config
        self.mode = mode
        self.log_freq = log_freq

        self._run = None
        self._initialized = False

    def on_train_begin(self, engine) -> None:
        """初始化 WandB"""
        if self._initialized:
            return

        self._run = wandb.init(
            project=self.project,
            name=self.name,
            config=self.config,
            mode=self.mode,
            reinit=True,
        )
        self._initialized = True

    def on_batch_end(self, engine, batch: int, logs: dict) -> None:
        """记录 batch 指标"""
        if batch % self.log_freq != 0:
            return

        wandb.log({
            "train/loss": logs.get("loss"),
            "train/lr": engine.lr,
            "train/epoch": engine.current_epoch,
            "train/step": engine.global_step,
        })

    def on_epoch_end(self, engine, epoch: int, logs: dict) -> None:
        """记录 epoch 指标"""
        wandb.log({
            "val/psnr": logs.get("val_psnr"),
            "val/ssim": logs.get("val_ssim"),
            "val/mae": logs.get("val_mae"),
            "val/max_mae": logs.get("val_max_mae"),
            "val/loss": logs.get("train_loss"),
            "val/lr": logs.get("lr"),
            "epoch": epoch,
        })

    def on_train_end(self, engine) -> None:
        """结束 WandB run"""
        if self._run is not None:
            wandb.finish()
            self._run = None
            self._initialized = False