from __future__ import annotations

import numpy as np
import torch
from torchvision.transforms.functional import InterpolationMode, resize


def combine_wind_speed(u10: np.ndarray, v10: np.ndarray) -> np.ndarray:
    if u10.shape != v10.shape:
        raise ValueError(f"u10 and v10 shapes differ: {tuple(u10.shape)} vs {tuple(v10.shape)}")
    return np.nan_to_num(np.hypot(u10, v10), nan=0.0).astype(np.float32, copy=False)


def wave_first_mask(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values[0])


def fill_nan(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(values, nan=0.0).astype(np.float32, copy=False)


def encode_mwd(mwd_degrees: np.ndarray) -> np.ndarray:
    values = fill_nan(mwd_degrees)
    angle = np.deg2rad(values)
    return np.stack([np.cos(angle), np.sin(angle)]).astype(np.float32, copy=False)


def trim_to_scale(array: np.ndarray, scale: int) -> np.ndarray:
    height, width = array.shape[-2:]
    trimmed_height = height - height % scale
    trimmed_width = width - width % scale
    return array[..., :trimmed_height, :trimmed_width]


def bicubic_downsample(array: np.ndarray, scale: int) -> np.ndarray:
    tensor = torch.from_numpy(array)
    if tensor.ndim == 2:
        tensor = tensor[None, ...]
    height, width = tensor.shape[-2:]
    lr = resize(
        tensor,
        [height // scale, width // scale],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    lr = lr[0] if array.ndim == 2 else lr
    return lr.to(dtype=torch.float32).numpy()
