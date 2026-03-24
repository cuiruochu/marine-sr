"""检查点回调"""

from pathlib import Path
from typing import Literal

from src.core.callbacks import Callback


class CheckpointCallback(Callback):
    """模型保存回调"""

    def __init__(
        self,
        save_dir: str,
        every: int = 10,
        save_best: bool = True,
        monitor: str = "val_loss",
        mode: Literal["min", "max"] = "min",
        keep_last: int = 0,
    ):
        self.save_dir = Path(save_dir)
        self.every = every
        self.save_best = save_best
        self.monitor = monitor
        self.mode = mode
        self.keep_last = keep_last

        if mode == "min":
            self.best_value = float("inf")
            self.is_better = lambda current, best: current < best
        else:
            self.best_value = float("-inf")
            self.is_better = lambda current, best: current > best

        self._saved_epochs: list = []

    def on_train_begin(self, engine) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, engine, epoch: int, logs: dict) -> None:
        if self.every > 0 and epoch % self.every == 0:
            path = self.save_dir / f"epoch_{epoch}.pth"
            self._saved_epochs.append(epoch)
            old_path = None

            if self.keep_last > 0 and len(self._saved_epochs) > self.keep_last:
                old_epoch = self._saved_epochs.pop(0)
                old_path = self.save_dir / f"epoch_{old_epoch}.pth"

            engine.save_checkpoint(str(path), epoch)

            if old_path is not None and old_path.exists():
                old_path.unlink()

        if self.save_best and self.monitor in logs:
            current = logs[self.monitor]
            if self.is_better(current, self.best_value):
                self.best_value = current
                engine.best_metric = current
                engine.best_epoch = epoch

                path = self.save_dir / "best.pth"
                engine.save_checkpoint(str(path), epoch)

    def on_train_end(self, engine) -> None:
        path = self.save_dir / "last.pth"
        engine.save_checkpoint(str(path), engine.current_epoch)

    def state_dict(self) -> dict:
        return {
            "best_value": self.best_value,
            "saved_epochs": list(self._saved_epochs),
        }

    def load_state_dict(self, state: dict) -> None:
        if "best_value" in state:
            self.best_value = state["best_value"]
        if "saved_epochs" in state:
            self._saved_epochs = list(state["saved_epochs"])
