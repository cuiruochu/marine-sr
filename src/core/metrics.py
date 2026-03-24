"""指标计算函数。"""

from typing import Optional

import torch
import torch.nn.functional as F

EPS = 1e-12


def _reduction_dims(img: torch.Tensor) -> tuple[int, ...]:
    if img.ndim < 2:
        raise ValueError(f"指标输入至少需要 2 维，当前 shape={tuple(img.shape)}")
    return tuple(range(1, img.ndim))


def _build_gaussian_window(
    kernel_size: int,
    sigma: float,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    gaussian = torch.exp(-(coords**2) / (2 * sigma**2))
    gaussian = gaussian / gaussian.sum()
    window_2d = torch.outer(gaussian, gaussian)
    window = window_2d.unsqueeze(0).unsqueeze(0)
    return window.expand(channels, 1, kernel_size, kernel_size).contiguous()


def _broadcast_mask(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    return torch.broadcast_to(mask.to(device=target.device), target.shape).to(dtype=target.dtype)


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """计算 PSNR，假设输入已经在 [0, 1] 范围内。"""
    dims = _reduction_dims(img1)
    squared_error = (img1 - img2) ** 2

    if mask is not None:
        mask = _broadcast_mask(mask, img1)
        mse = (squared_error * mask).sum(dim=dims) / mask.sum(dim=dims).clamp_min(1.0)
    else:
        mse = squared_error.mean(dim=dims)

    psnr = torch.where(mse <= EPS, torch.full_like(mse, float("inf")), 10.0 * torch.log10(1.0 / mse.clamp_min(EPS)))
    return psnr.mean().item()


def calculate_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    kernel_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """计算窗口化 SSIM，假设输入已经在 [0, 1] 范围内。"""
    if img1.ndim != 4 or img2.ndim != 4:
        raise ValueError(f"SSIM 仅支持 NCHW 四维输入，当前 img1={tuple(img1.shape)}, img2={tuple(img2.shape)}")
    if img1.shape != img2.shape:
        raise ValueError(f"SSIM 输入 shape 必须一致: img1={tuple(img1.shape)}, img2={tuple(img2.shape)}")
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size 必须为奇数，当前={kernel_size}")

    channels = img1.shape[1]
    padding = kernel_size // 2
    window = _build_gaussian_window(kernel_size, sigma, channels, img1.device, img1.dtype)

    if mask is None:
        mask = torch.ones_like(img1)
    else:
        mask = _broadcast_mask(mask, img1)

    mask_map = F.conv2d(mask, window, padding=padding, groups=channels)
    safe_mask_map = mask_map.clamp_min(EPS)

    mu1 = F.conv2d(img1 * mask, window, padding=padding, groups=channels) / safe_mask_map
    mu2 = F.conv2d(img2 * mask, window, padding=padding, groups=channels) / safe_mask_map
    second1 = F.conv2d(img1 * img1 * mask, window, padding=padding, groups=channels) / safe_mask_map
    second2 = F.conv2d(img2 * img2 * mask, window, padding=padding, groups=channels) / safe_mask_map
    second12 = F.conv2d(img1 * img2 * mask, window, padding=padding, groups=channels) / safe_mask_map

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (second1 - mu1_sq).clamp_min(0.0)
    sigma2_sq = (second2 - mu2_sq).clamp_min(0.0)
    sigma12 = second12 - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim = numerator / denominator.clamp_min(EPS)
    valid = mask_map > 0
    if valid.any():
        return ssim.masked_select(valid).mean().item()
    return 0.0


def calculate_mae(img1: torch.Tensor, img2: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """计算 MAE。"""
    dims = _reduction_dims(img1)
    diff = torch.abs(img1 - img2)

    if mask is not None:
        mask = _broadcast_mask(mask, img1)
        mae = (diff * mask).sum(dim=dims) / mask.sum(dim=dims).clamp_min(1.0)
    else:
        mae = diff.mean(dim=dims)
    return mae.mean().item()


def calculate_max_mae(img1: torch.Tensor, img2: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """计算最大 MAE。"""
    dims = _reduction_dims(img1)
    diff = torch.abs(img1 - img2)
    if mask is not None:
        mask = _broadcast_mask(mask, img1)
        diff = diff * mask
    return diff.amax(dim=dims).max().item()


def reverse_norm(img: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """反归一化。"""
    return img * std + mean


def normalize_to_01(img1: torch.Tensor, img2: torch.Tensor) -> tuple:
    """归一化到 [0, 1] 范围用于 PSNR/SSIM 计算。"""
    dims = _reduction_dims(img1)
    min_val = torch.minimum(
        img1.amin(dim=dims, keepdim=True),
        img2.amin(dim=dims, keepdim=True),
    )
    max_val = torch.maximum(
        img1.amax(dim=dims, keepdim=True),
        img2.amax(dim=dims, keepdim=True),
    )
    scale = max_val - min_val
    safe_scale = scale.clamp_min(1e-8)

    img1_norm = (img1 - min_val) / safe_scale
    img2_norm = (img2 - min_val) / safe_scale
    zero_scale = scale < 1e-8
    img1_norm = torch.where(zero_scale, torch.zeros_like(img1_norm), img1_norm)
    img2_norm = torch.where(zero_scale, torch.zeros_like(img2_norm), img2_norm)

    return img1_norm, img2_norm


def apply_output_mask(sr: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """将输出结果应用 mask，False 区域置零，后续保存时再转成 None。"""
    return sr * _broadcast_mask(mask, sr)
