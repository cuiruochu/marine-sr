"""评估引擎。"""

import logging
from typing import Any

import numpy as np
import torch
from torch import nn

from .callbacks import Callback, CallbackList
from .metrics import (
    apply_output_mask,
    calculate_mae,
    calculate_max_mae,
    calculate_psnr,
    calculate_ssim,
    normalize_to_01,
    reverse_norm,
)
from .model_output import normalize_model_output
from .progress import build_progress_points

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(
        self,
        model: nn.Module,
        device: str | None = None,
        callbacks: list[Callback] | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.callbacks = CallbackList(callbacks or [])
        self.current_batch = 0
        self.total_batches = 0

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.no_grad()
    def run(
        self,
        test_loader: Any,
        mean: list[float],
        std: list[float],
        mask: np.ndarray | None = None,
        mode: str = "evaluation",
    ) -> dict[str, float]:
        self.model.eval()
        self.total_batches = len(test_loader)
        self.current_batch = 0
        progress_points = build_progress_points(self.total_batches)
        stage_name = "Evaluation" if mode == "evaluation" else "Inference"

        mean_t = torch.tensor(mean, device=self.device).view(-1, 1, 1)
        std_t = torch.tensor(std, device=self.device).view(-1, 1, 1)
        mask_t = None
        if mask is not None:
            mask_t = torch.from_numpy(mask).to(self.device)

        self.callbacks.on_eval_begin(self)
        logger.info(f"{stage_name} started: batches={self.total_batches}")

        total_psnr, total_ssim, total_mae, total_max_mae = 0.0, 0.0, 0.0, 0.0
        metric_samples = 0

        for batch in test_loader:
            self.current_batch += 1
            self.callbacks.on_batch_begin(self, self.current_batch)

            sr, hr, filename = self._eval_step(batch, mean_t, std_t)

            if hr is None:
                self._validate_mask_shape(mask_t, sr)
                sr_for_save = apply_output_mask(sr, mask_t) if mask_t is not None else sr
                self.callbacks.on_batch_end(
                    self,
                    self.current_batch,
                    sr=sr_for_save,
                    hr=None,
                    filename=filename,
                    metrics=None,
                    mask=mask,
                )
                if self.current_batch in progress_points:
                    progress = int(self.current_batch * 100 / self.total_batches)
                    logger.info(
                        f"{stage_name} progress: {self.current_batch}/{self.total_batches} ({progress}%)"
                    )
                continue

            # PSNR/SSIM are defined on tensors normalized to [0, 1].
            self._validate_mask_shape(mask_t, hr)
            hr_norm, sr_norm = normalize_to_01(hr, sr)

            batch_psnr = calculate_psnr(hr_norm, sr_norm, mask=mask_t)
            batch_ssim = calculate_ssim(hr_norm, sr_norm, mask=mask_t)
            batch_mae = calculate_mae(sr, hr, mask=mask_t)
            batch_max_mae = calculate_max_mae(sr, hr, mask=mask_t)
            batch_size = hr.shape[0]

            total_psnr += batch_psnr * batch_size
            total_ssim += batch_ssim * batch_size
            total_mae += batch_mae * batch_size
            total_max_mae = max(total_max_mae, batch_max_mae)
            metric_samples += batch_size

            metrics = {
                "psnr": batch_psnr,
                "ssim": batch_ssim,
                "mae": batch_mae,
                "max_mae": batch_max_mae,
            }
            sr_for_save = apply_output_mask(sr, mask_t) if mask_t is not None else sr
            self.callbacks.on_batch_end(
                self,
                self.current_batch,
                sr=sr_for_save,
                hr=hr,
                filename=filename,
                metrics=metrics,
                mask=mask,
            )
            if self.current_batch in progress_points:
                progress = int(self.current_batch * 100 / self.total_batches)
                logger.info(
                    f"{stage_name} progress: {self.current_batch}/{self.total_batches} ({progress}%)"
                )

        metrics = {}
        if metric_samples > 0:
            metrics = {
                "psnr": total_psnr / metric_samples,
                "ssim": total_ssim / metric_samples,
                "mae": total_mae / metric_samples,
                "max_mae": total_max_mae,
            }

        self.callbacks.on_eval_end(self, metrics)
        return metrics

    def _eval_step(
        self,
        batch: Any,
        mean_t: torch.Tensor,
        std_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[str]]:
        if len(batch) == 3:
            lr, hr, filename = batch[0].to(self.device), batch[1].to(self.device), batch[2]
        elif len(batch) == 2:
            lr, filename = batch[0].to(self.device), batch[1]
            hr = None
        else:
            raise ValueError(f"不支持的评估 batch 结构，长度={len(batch)}")

        sr = normalize_model_output(self.model(lr)).pred
        sr = reverse_norm(sr, mean_t, std_t)

        filenames = self._normalize_filenames(filename, batch_size=lr.shape[0])
        return sr, hr, filenames

    @staticmethod
    def _normalize_filenames(filename: Any, batch_size: int) -> list[str]:
        if isinstance(filename, str):
            filenames = [filename]
        elif isinstance(filename, (list, tuple)):
            filenames = [str(item) for item in filename]
        else:
            raise TypeError(f"filename 必须是字符串或字符串列表，当前类型={type(filename)}")

        if len(filenames) != batch_size:
            raise ValueError(f"filename 数量必须与 batch_size 一致，当前 filenames={len(filenames)}，batch_size={batch_size}")

        return filenames

    @staticmethod
    def _validate_mask_shape(mask: torch.Tensor | None, target: torch.Tensor) -> None:
        if mask is None:
            return
        expected = tuple(target.shape[-2:])
        if tuple(mask.shape[-2:]) != expected:
            raise ValueError(f"mask shape 必须等于目标 H,W，期望={expected}，当前={tuple(mask.shape[-2:])}")
