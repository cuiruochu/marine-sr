"""数据集共享辅助。"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data

from src.utils.data_files import build_paired_npy_files, list_npy_files

from .utils import patchify


def to_chw_tensor(array: np.ndarray) -> torch.Tensor:
    tensor = torch.tensor(array, dtype=torch.float32)
    if tensor.ndim == 2:
        return tensor.unsqueeze(0)
    if tensor.ndim == 3:
        return tensor
    raise ValueError(f"仅支持二维(H, W)或三维(C, H, W)数组，当前形状={tuple(tensor.shape)}")


def infer_channel_count(path: Path) -> int:
    array = np.load(path, mmap_mode="r")
    if array.ndim == 2:
        return 1
    if array.ndim == 3:
        return int(array.shape[0])
    raise ValueError(f"不支持的样本维度: {path} shape={tuple(array.shape)}")


def build_paired_file_list(lr_root: str, hr_root: str) -> list[tuple[str, str]]:
    return [(str(lr_path), str(hr_path)) for lr_path, hr_path in build_paired_npy_files(lr_root, hr_root)]


def build_file_list(root: str) -> list[str]:
    return [str(path) for path in list_npy_files(root)]


def paired_patchify(
    lr: torch.Tensor,
    hr: torch.Tensor,
    lr_patch_size: int,
    hr_patch_size: int,
    upscale: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    lr_patch, x0, y0 = patchify(lr, lr_patch_size, return_coords=True)
    hr_x0 = x0 * upscale
    hr_y0 = y0 * upscale
    hr_patch = hr[:, hr_x0 : hr_x0 + hr_patch_size, hr_y0 : hr_y0 + hr_patch_size]
    return lr_patch, hr_patch


class BaseMarineDataset(data.Dataset):
    sample_pairs: list[tuple[str, str]]

    def __init__(self, *, upscale: int, mean: list[float], std: list[float]):
        self.upscale = upscale
        self.mean = torch.tensor(mean, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
        self.std = torch.tensor(std, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
        if not hasattr(self, "sample_pairs"):
            raise NotImplementedError("子类必须先初始化 sample_pairs")
        if not self.sample_pairs:
            raise ValueError("sample_pairs 不能为空")

        self.channel_count = infer_channel_count(Path(self.sample_pairs[0][0]))
        self.channel_counts = [self.channel_count] * len(self.sample_pairs)

    def __len__(self):
        return len(self.sample_pairs)

    def load_file(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        lr_path, hr_path = self.sample_pairs[index]
        lr = to_chw_tensor(np.load(lr_path))
        hr = to_chw_tensor(np.load(hr_path))
        if lr.shape[0] != self.channel_count or hr.shape[0] != self.channel_count:
            raise ValueError(
                f"样本通道数不一致: lr={lr_path} shape={tuple(lr.shape)}, hr={hr_path} shape={tuple(hr.shape)}"
            )
        return lr, hr

    def get_filename(self, index: int) -> str:
        return os.path.basename(self.sample_pairs[index][1])
