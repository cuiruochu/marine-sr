"""
指标计算模块

提供 PSNR、SSIM、MAE 等评估指标的计算函数。
"""

import torch
import numpy as np
import torch.nn.functional as F
import cv2
from typing import Optional, Tuple, Union


def reverse_norm(
    img: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor
) -> torch.Tensor:
    """
    反归一化

    Args:
        img: 图像张量 (B, C, H, W) 或 (C, H, W)
        mean: 均值 (C, 1, 1)
        std: 标准差 (C, 1, 1)

    Returns:
        反归一化后的图像
    """
    return img * std + mean


def normalize_to_01(
    hr_img: torch.Tensor,
    hr_hat: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将 HR 和 SR 归一化到 [0, 1] 范围

    使用两者的联合最小最大值进行归一化。

    Args:
        hr_img: HR 图像 (B, C, H, W)
        hr_hat: SR 图像 (B, C, H, W)

    Returns:
        (hr_norm, sr_norm): 归一化后的图像
    """
    def norm(tensor, min_val, max_val):
        return (tensor - min_val) / (max_val - min_val + 1e-8)

    min_vals = torch.minimum(
        hr_img.view(hr_img.size(0), -1).min(dim=1).values,
        hr_hat.view(hr_hat.size(0), -1).min(dim=1).values
    )
    max_vals = torch.maximum(
        hr_img.view(hr_img.size(0), -1).max(dim=1).values,
        hr_hat.view(hr_hat.size(0), -1).max(dim=1).values
    )

    min_vals = min_vals.view(-1, 1, 1, 1)
    max_vals = max_vals.view(-1, 1, 1, 1)

    hr_norm = norm(hr_img, min_vals, max_vals)
    sr_norm = norm(hr_hat, min_vals, max_vals)

    return hr_norm, sr_norm


def apply_mask(
    mask: Optional[torch.Tensor],
    *imgs: torch.Tensor
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """
    应用掩码到图像

    Args:
        mask: 掩码张量，None 表示不应用
        *imgs: 图像张量

    Returns:
        应用掩码后的图像（单个或元组）
    """
    res = []
    for img in imgs:
        if mask is not None:
            if mask.shape != img.shape:
                mask = _expand_mask(mask, img)
            img = img * mask
        res.append(img)
    return res[0] if len(res) == 1 else tuple(res)


def calculate_mae(
    img: torch.Tensor,
    img_hat: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> float:
    """
    计算 MAE (Mean Absolute Error)

    Args:
        img: 真实图像
        img_hat: 预测图像
        mask: 可选掩码

    Returns:
        MAE 值
    """
    loss_map = torch.abs(img - img_hat)
    if mask is not None:
        mask = _expand_mask(mask, loss_map)
        loss_map = loss_map[mask]
    return loss_map.mean().item()


def calculate_max_mae(
    img: torch.Tensor,
    img_hat: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> float:
    """
    计算最大 MAE

    Args:
        img: 真实图像
        img_hat: 预测图像
        mask: 可选掩码

    Returns:
        最大 MAE 值
    """
    loss_map = torch.abs(img - img_hat)
    if mask is not None:
        mask = _expand_mask(mask, loss_map)
        loss_map = loss_map[mask]
    return loss_map.max().item()


def calculate_psnr(
    img: torch.Tensor,
    img_hat: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> float:
    """
    计算 PSNR (Peak Signal-to-Noise Ratio)

    注意：输入图像应已归一化到 [0, 1] 范围

    Args:
        img: 真实图像 (已归一化)
        img_hat: 预测图像 (已归一化)
        mask: 可选掩码

    Returns:
        PSNR 值 (dB)
    """
    loss_map = (img - img_hat) ** 2
    if mask is not None:
        mask = _expand_mask(mask, loss_map)
        loss_map = loss_map[mask]
    mse = loss_map.mean()
    psnr = 10. * torch.log10(1. / (mse + 1e-8))
    return psnr.item()


def calculate_ssim(
    img: torch.Tensor,
    img2: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> float:
    """
    计算 SSIM (Structural Similarity Index)

    注意：输入图像应已归一化到 [0, 1] 范围

    Args:
        img: 真实图像 (B, C, H, W)，已归一化
        img2: 预测图像 (B, C, H, W)，已归一化
        mask: 可选掩码 (H, W)，1 表示有效区域

    Returns:
        SSIM 值
    """
    if mask is None:
        mask = torch.ones_like(img)
    else:
        mask = _expand_mask(mask, img)

    # 转换数据类型
    img = img.to(torch.float64)
    img2 = img2.to(torch.float64)
    mask = mask.to(img.dtype)

    # 转换到 [0, 255] 范围
    img = img * 255.
    img2 = img2 * 255.

    # SSIM 常数
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    # 创建高斯窗口
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    window = torch.from_numpy(window).view(1, 1, 11, 11).expand(
        img.size(1), 1, 11, 11
    ).to(img.dtype).to(img.device)

    # 卷积参数
    conv_params = dict(stride=1, padding=0, groups=img.shape[1])
    eps = 1e-12

    # 计算每个窗口的权重和
    W_sum = F.conv2d(mask, window, **conv_params)

    # 计算加权均值
    mu1 = F.conv2d(img * mask, window, **conv_params) / (W_sum + eps)
    mu2 = F.conv2d(img2 * mask, window, **conv_params) / (W_sum + eps)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # 计算加权方差和协方差
    sigma1_sq = F.conv2d(img * img * mask, window, **conv_params) / (W_sum + eps) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2 * mask, window, **conv_params) / (W_sum + eps) - mu2_sq
    sigma12 = F.conv2d(img * img2 * mask, window, **conv_params) / (W_sum + eps) - mu1_mu2

    # SSIM map
    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map

    # 计算加权平均 SSIM
    sum_W_sum = torch.sum(W_sum, dim=[1, 2, 3])
    ssim_per_image = torch.sum(ssim_map * W_sum, dim=[1, 2, 3]) / (sum_W_sum + eps)
    ssim_per_image[sum_W_sum == 0] = 0

    return ssim_per_image.mean().item()


def _expand_mask(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if mask.shape == reference.shape:
        return mask
    if mask.ndim != reference.ndim:
        raise ValueError(f"mask 维度与参考张量不一致: mask={mask.shape}, ref={reference.shape}")
    if mask.size(0) != reference.size(0):
        raise ValueError(f"mask batch 维度与参考张量不一致: mask={mask.shape}, ref={reference.shape}")
    if mask.size(1) == 1 and reference.size(1) > 1:
        return mask.expand(-1, reference.size(1), -1, -1)
    raise ValueError(f"mask 形状无法广播到参考张量: mask={mask.shape}, ref={reference.shape}")
