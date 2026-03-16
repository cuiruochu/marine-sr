import torch
import numpy as np
import torch.nn.functional as F
import cv2


def reverse_norm(img, mean, std):
    """
    :param img: (C, H, W)
    :param mean: (C, 1, 1)
    :param std: (C, 1, 1)
    """
    return img * std + mean


def normalize_to_01(hr_img, hr_hat):
    """Normalize tensor to [0, 1] range using min and max values."""

    def norm(tensor, min_val, max_val):
        return (tensor - min_val) / (max_val - min_val)

    min_vals = torch.minimum(
        hr_img.view(hr_img.size(0), -1).min(dim=1).values,  # Min of hr
        hr_hat.view(hr_hat.size(0), -1).min(dim=1).values  # Min of hr_hat
    )  # Shape: (B,)
    max_vals = torch.maximum(
        hr_img.view(hr_img.size(0), -1).max(dim=1).values,  # Max of hr
        hr_hat.view(hr_hat.size(0), -1).max(dim=1).values  # Max of hr_hat
    )  # Shape: (B,)
    # Reshape min and max values to (B, 1, 1, 1) for broadcasting
    min_vals = min_vals.view(-1, 1, 1, 1)
    max_vals = max_vals.view(-1, 1, 1, 1)

    # Normalize hr and hr_hat to [0, 1] range using joint min and max
    hr_norm = norm(hr_img, min_vals, max_vals)
    hr_hat_norm = norm(hr_hat, min_vals, max_vals)

    return hr_norm, hr_hat_norm


def apply_mask(mask, *imgs):
    res = []
    for img in imgs:
        if mask is not None:
            img = img * mask
        res.append(img)
    return res[0] if len(res) == 1 else tuple(res)


def calculate_mae_pt(img, img_hat, mask=None):
    loss_map = torch.abs(img - img_hat)
    if mask is not None:
        loss_map = loss_map[mask]
    return loss_map.mean()


def calculate_max_mae_pt(img, img_hat, mask=None):
    loss_map = torch.abs(img - img_hat)
    if mask is not None:
        loss_map = loss_map[mask]
    return loss_map.max()


def calculate_psnr_pt(img, img_hat, mask=None):
    loss_map = (img - img_hat) ** 2
    if mask is not None:
        loss_map = loss_map[mask]
    mse = loss_map.mean()
    psnr = 10. * torch.log10(1. / (mse + 1e-8))
    return psnr


def calculate_ssim_pt(img, img2, mask=None):
    """确保已经归一化

    Args:
        img (Tensor): Images with range [0, 255], shape (n, c, h, w).
        img2 (Tensor): Images with range [0, 255], shape (n, c, h, w).
        mask (Tensor): Mask with range [0, 1], shape (h, w).
                       1 for valid region, 0 for masked region.

    Returns:
        float: SSIM result.
    """

    if mask is None:
        mask = torch.ones_like(img)

    # img范围
    img = img.to(torch.float64)
    img2 = img2.to(torch.float64)
    mask = mask.to(img.dtype)
    img, img2 = img * 255., img2 * 255.

    # SSIM 常数
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    # 创建高斯窗口
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    window = torch.from_numpy(window).view(1, 1, 11, 11).expand(img.size(1), 1, 11, 11).to(img.dtype).to(img.device)

    # 卷积参数
    conv_params = dict(stride=1, padding=0, groups=img.shape[1])

    # 1. 计算每个窗口的权重和 (W_sum)
    # epsilon来避免除以零
    eps = 1e-12
    W_sum = F.conv2d(mask, window, **conv_params)
    # 2. 计算加权均值 (μ)
    mu1 = F.conv2d(img * mask, window, **conv_params) / (W_sum + eps)
    mu2 = F.conv2d(img2 * mask, window, **conv_params) / (W_sum + eps)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    # 3. 计算加权方差 (σ²) 和协方差 (σ_xy)
    sigma1_sq = F.conv2d(img * img * mask, window, **conv_params) / (W_sum + eps) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2 * mask, window, **conv_params) / (W_sum + eps) - mu2_sq
    sigma12 = F.conv2d(img * img2 * mask, window, **conv_params) / (W_sum + eps) - mu1_mu2
    # SSIM map
    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map

    # 4. 计算每个图像的加权平均SSIM
    sum_W_sum = torch.sum(W_sum, dim=[1, 2, 3])
    ssim_per_image = torch.sum(ssim_map * W_sum, dim=[1, 2, 3]) / (sum_W_sum + eps)
    # 对于那些W_sum完全为0的图像（即mask全黑），其ssim将为0
    ssim_per_image[sum_W_sum == 0] = 0
    return ssim_per_image.mean()
