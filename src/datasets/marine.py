"""Marine 数据集实现。"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .common import (
    BaseMarineDataset,
    build_file_list,
    build_paired_file_list,
    infer_channel_count,
    paired_patchify,
    to_chw_tensor,
)
from .utils import augement, normalize


class MarineTrainDataset(BaseMarineDataset):
    """训练数据集：配对 patch -> 增强 -> z-score。"""

    def __init__(
        self,
        lr_root: str,
        hr_root: str,
        *,
        upscale: int,
        lr_patch_size: int,
        mean: list[float],
        std: list[float],
    ):
        self.sample_pairs = build_paired_file_list(lr_root, hr_root)
        self.lr_patch_size = lr_patch_size
        self.hr_patch_size = int(lr_patch_size * upscale)
        super().__init__(upscale=upscale, mean=mean, std=std)

    def __getitem__(self, index):
        lr, hr = self.load_file(index)
        lr_patch, hr_patch = paired_patchify(lr, hr, self.lr_patch_size, self.hr_patch_size, self.upscale)
        lr_patch, hr_patch = augement(lr_patch, hr_patch)
        lr_patch, hr_patch = normalize(self.mean, self.std, lr_patch, hr_patch)
        return lr_patch.contiguous(), hr_patch.contiguous()


class MarineEvalDataset(BaseMarineDataset):
    """评估/验证数据集。可选是否对 HR 也做 z-score。"""

    def __init__(
        self,
        lr_root: str,
        hr_root: str,
        *,
        upscale: int,
        mean: list[float],
        std: list[float],
        sample_limit: int | bool = False,
        return_filename: bool = True,
        normalize_hr: bool = False,
    ):
        sample_pairs = build_paired_file_list(lr_root, hr_root)
        self.sample_pairs = sample_pairs[: int(sample_limit)] if sample_limit else sample_pairs
        self.return_filename = return_filename
        self.normalize_hr = normalize_hr
        super().__init__(upscale=upscale, mean=mean, std=std)

    def __getitem__(self, index):
        lr, hr = self.load_file(index)
        lr = normalize(self.mean, self.std, lr)
        if self.normalize_hr:
            hr = normalize(self.mean, self.std, hr)
        if self.return_filename:
            return lr.contiguous(), hr.contiguous(), self.get_filename(index)
        return lr.contiguous(), hr.contiguous()


class MarineInferDataset(Dataset):
    """推理数据集：仅加载 LR 并做 z-score。"""

    def __init__(
        self,
        lr_root: str,
        *,
        upscale: int,
        mean: list[float],
        std: list[float],
        sample_limit: int | bool = False,
        return_filename: bool = True,
    ):
        self.sample_files = build_file_list(lr_root)
        if sample_limit:
            self.sample_files = self.sample_files[: int(sample_limit)]
        if not self.sample_files:
            raise ValueError("推理数据集不能为空")

        self.upscale = upscale
        self.mean = torch.tensor(mean, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
        self.std = torch.tensor(std, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
        self.return_filename = return_filename
        self.channel_count = infer_channel_count(self.sample_files[0])
        self.channel_counts = [self.channel_count] * len(self.sample_files)

    def __len__(self) -> int:
        return len(self.sample_files)

    def __getitem__(self, index):
        lr_path = self.sample_files[index]
        lr = to_chw_tensor(np.load(lr_path))
        if lr.shape[0] != self.channel_count:
            raise ValueError(f"样本通道数不一致: {lr_path} shape={tuple(lr.shape)}")
        lr = normalize(self.mean, self.std, lr)
        filename = os.path.basename(lr_path)
        if self.return_filename:
            return lr.contiguous(), filename
        return lr.contiguous()
