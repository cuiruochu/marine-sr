"""Wandb 回调"""

from typing import Any, Dict, Optional

from src.core.callbacks import Callback


class WandbCallback(Callback):
    """Wandb 日志回调"""

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        mode: str = "offline",
    ):
        self.project = project
        self.name = name
        self.config = config
        self.mode = mode
        self._run = None

    def on_train_begin(self, engine) -> None:
        import wandb

        self._run = wandb.init(
            project=self.project,
            name=self.name,
            config=self.config,
            mode=self.mode,
        )

    def on_epoch_end(self, engine, epoch: int, logs: dict) -> None:
        import wandb

        wandb.log(logs)

    def on_train_end(self, engine) -> None:
        import wandb

        if self._run is not None:
            wandb.finish()
