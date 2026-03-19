"""
海洋参数超分辨率数据集定义
"""

import os

import numpy as np
import torch
import torch.utils.data as data

from src.utils.data_files import build_paired_npy_files, list_npy_files

from .utils import augement, normalize, patchify


class BaseMarineDataset(data.Dataset):
    sample_pairs: list[tuple[str, str]]

    def __init__(self, upscale: int, mean: list[float], std: list[float]):
        self.upscale = upscale
        self.mean = torch.tensor(mean).unsqueeze(-1).unsqueeze(-1)
        self.std = torch.tensor(std).unsqueeze(-1).unsqueeze(-1)
        if not hasattr(self, "sample_pairs"):
            raise NotImplementedError("子类必须在 super() 之前初始化 sample_pairs")

    def load_file(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        lr_path, hr_path = self.sample_pairs[index]
        lr = np.load(lr_path)
        hr = np.load(hr_path)
        lr = _to_chw_tensor(lr)
        hr = _to_chw_tensor(hr)
        return lr, hr

    def __len__(self) -> int:
        return len(self.sample_pairs)

    def get_filename(self, index: int) -> str:
        return os.path.basename(self.sample_pairs[index][1])


class MarineEvalDataset(BaseMarineDataset):
    """评估/验证数据集"""

    def __init__(
        self,
        lr_root: str,
        hr_root: str,
        upscale: int,
        mean: list[float],
        std: list[float],
        max_sample: int | bool = False,
    ):
        sample_pairs = _build_paired_file_list(lr_root, hr_root)
        self.sample_pairs = sample_pairs[:max_sample] if max_sample else sample_pairs
        super().__init__(upscale, mean, std)

    def __getitem__(self, index: int):
        lr, hr = self.load_file(index)
        lr = normalize(self.mean, self.std, lr)
        filename = self.get_filename(index)
        return lr.contiguous(), hr.contiguous(), filename


class MarineInferenceDataset(data.Dataset):
    """离线推理数据集，仅加载 LR"""

    def __init__(
        self,
        lr_root: str,
        mean: list[float],
        std: list[float],
    ):
        self.lr_list = [str(path) for path in list_npy_files(lr_root)]
        self.mean = torch.tensor(mean).unsqueeze(-1).unsqueeze(-1)
        self.std = torch.tensor(std).unsqueeze(-1).unsqueeze(-1)

    def __len__(self) -> int:
        return len(self.lr_list)

    def __getitem__(self, index: int):
        lr = np.load(self.lr_list[index])
        lr = _to_chw_tensor(lr)
        lr = normalize(self.mean, self.std, lr)
        filename = os.path.basename(self.lr_list[index])
        return lr.contiguous(), filename


class MarineTrainDataset(BaseMarineDataset):
    """训练数据集"""

    def __init__(
        self,
        lr_root: str,
        hr_root: str,
        upscale: int,
        lr_patch_size: int,
        mean: list[float],
        std: list[float],
    ):
        self.sample_pairs = _build_paired_file_list(lr_root, hr_root)
        self.hr_patch_size = int(lr_patch_size * upscale)
        self.lr_patch_size = lr_patch_size
        super().__init__(upscale, mean, std)

    def __getitem__(self, index: int):
        lr, hr = self.load_file(index)
        lr, hr = _paired_patchify(lr, hr, self.lr_patch_size, self.hr_patch_size, self.upscale)
        lr, hr = augement(lr, hr)
        lr, hr = normalize(self.mean, self.std, lr, hr)
        return lr.contiguous(), hr.contiguous()


def _build_paired_file_list(lr_root: str, hr_root: str) -> list[tuple[str, str]]:
    return [(str(lr_path), str(hr_path)) for lr_path, hr_path in build_paired_npy_files(lr_root, hr_root)]


def _paired_patchify(
    lr: torch.Tensor,
    hr: torch.Tensor,
    lr_patch_size: int,
    hr_patch_size: int,
    upscale: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    lr_patch, x0, y0 = patchify(lr, lr_patch_size, return_coords=True)
    hr_x0 = x0 * upscale
    hr_y0 = y0 * upscale
    hr_patch = hr[:, hr_x0: hr_x0 + hr_patch_size, hr_y0: hr_y0 + hr_patch_size]
    return lr_patch, hr_patch


def _to_chw_tensor(array: np.ndarray) -> torch.Tensor:
    tensor = torch.tensor(array, dtype=torch.float32)
    if tensor.ndim == 2:
        return tensor.unsqueeze(0)
    if tensor.ndim == 3:
        return tensor
    raise ValueError(f"仅支持二维(H, W)或三维(C, H, W)数组，当前形状={tuple(tensor.shape)}")
