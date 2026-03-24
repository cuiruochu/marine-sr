"""训练引擎。"""

import logging
from typing import Any, Callable, Dict, List, Optional

import torch
from torch import nn
from torch.optim import Optimizer

from src.core.callbacks import Callback, CallbackList
from src.core.model_output import normalize_model_output, sum_aux_losses
from src.core.progress import build_progress_points
from src.utils.random_state import capture_rng_state, restore_rng_state

logger = logging.getLogger(__name__)


class Engine:
    """统一模型训练引擎。"""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: Callable,
        scheduler=None,
        device: Optional[str] = None,
        callbacks: Optional[List[Callback]] = None,
        config_snapshot: Optional[Dict[str, Any]] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.config_snapshot = config_snapshot

        self.callbacks = CallbackList(callbacks or [])

        self.current_epoch = 0
        self.global_step = 0

        self._train_loader = None
        self._val_loader = None

    @property
    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def train_step(self, batch: Any) -> Dict[str, Any]:
        """单个训练步骤。"""
        self.model.train()

        lr, hr = batch[0].to(self.device), batch[1].to(self.device)

        self.optimizer.zero_grad()

        output = normalize_model_output(self.model(lr))
        base_loss = self.loss_fn(output.pred, hr)
        aux_loss = sum_aux_losses(output.aux_losses, base_loss)
        loss = base_loss + aux_loss

        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "base_loss": base_loss.item(),
            "aux_loss": aux_loss.item(),
        }

    @torch.no_grad()
    def evaluate(self, val_loader: Any) -> Dict[str, float]:
        """在验证集上计算与训练一致的损失。"""
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            lr, hr = batch[0].to(self.device), batch[1].to(self.device)
            output = normalize_model_output(self.model(lr))
            base_loss = self.loss_fn(output.pred, hr)
            aux_loss = sum_aux_losses(output.aux_losses, base_loss)
            loss = base_loss + aux_loss

            total_loss += loss.item()
            num_batches += 1

        return {"val_loss": total_loss / num_batches}

    def fit(
        self,
        train_loader: Any,
        val_loader: Any,
        epochs: int,
        start_epoch: int = 1,
    ) -> Dict[str, Any]:
        """主训练循环。"""
        self._train_loader = train_loader
        self._val_loader = val_loader

        self.callbacks.on_train_begin(self)

        history = {"train_loss": [], "val_loss": []}

        for epoch in range(start_epoch, start_epoch + epochs):
            self.current_epoch = epoch

            self.callbacks.on_epoch_begin(self, epoch)

            epoch_loss = self._train_epoch(train_loader)
            val_logs = self.evaluate(val_loader)

            if self.scheduler is not None:
                self.scheduler.step(val_logs)

            val_logs["lr"] = self.lr
            val_logs["train_loss"] = epoch_loss

            self.callbacks.on_epoch_end(self, epoch, val_logs)

            history["train_loss"].append(epoch_loss)
            history["val_loss"].append(val_logs["val_loss"])

        self.callbacks.on_train_end(self)

        return history

    def _train_epoch(self, train_loader: Any) -> float:
        """训练一个 epoch。"""
        total_loss = 0.0
        num_batches = 0
        total_batches = len(train_loader)
        progress_points = build_progress_points(total_batches)
        logger.info(f"Epoch {self.current_epoch} started: batches={total_batches}, lr={self.lr:.2e}")

        for batch_idx, batch in enumerate(train_loader, 1):
            self.callbacks.on_batch_begin(self, batch_idx)

            logs = self.train_step(batch)
            self.global_step += 1

            self.callbacks.on_batch_end(self, batch_idx, logs)

            total_loss += logs["loss"]
            num_batches += 1

            if batch_idx in progress_points:
                progress = int(batch_idx * 100 / total_batches)
                logger.info(
                    f"Epoch {self.current_epoch} progress: "
                    f"{batch_idx}/{total_batches} ({progress}%), loss={logs['loss']:.4f}"
                )

        return total_loss / num_batches

    def save_checkpoint(self, path: str, epoch: Optional[int] = None) -> None:
        epoch = epoch or self.current_epoch

        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "callbacks": self.callbacks.state_dict(),
            "rng_state": capture_rng_state(),
        }

        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
        if self.config_snapshot is not None:
            state["config"] = self.config_snapshot

        torch.save(state, path)

    def load_checkpoint(
        self,
        path: str,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        load_callbacks: bool = True,
        load_rng_state: bool = True,
    ) -> int:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(ckpt["model"])

        if load_optimizer and "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])

        if load_scheduler and self.scheduler is not None and "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])

        self.current_epoch = ckpt.get("epoch", 0)
        self.global_step = ckpt.get("global_step", 0)
        if load_callbacks:
            self.callbacks.load_state_dict(ckpt.get("callbacks"))
        if load_rng_state:
            restore_rng_state(ckpt.get("rng_state"))

        return self.current_epoch

    def count_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.model.parameters())
