"""学习率监控回调。"""

from src.core.callbacks import Callback


class LRMonitorCallback(Callback):
    def __init__(self, log_freq: int = 100):
        self.log_freq = log_freq
        self.lr_history = []

    def on_batch_end(self, engine, batch: int, logs: dict | None = None, **kwargs) -> None:
        if batch % self.log_freq != 0:
            return

        for index, param_group in enumerate(engine.optimizer.param_groups):
            self.lr_history.append(
                {
                    "step": engine.global_step,
                    "epoch": engine.current_epoch,
                    "param_group": index,
                    "lr": param_group["lr"],
                }
            )

    def on_epoch_end(self, engine, epoch: int, logs: dict) -> None:
        logs["lr"] = engine.lr

    def get_lr_schedule(self):
        return self.lr_history.copy()
