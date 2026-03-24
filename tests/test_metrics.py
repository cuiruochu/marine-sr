"""
测试指标计算
"""

import pytest
import torch

from src.core.metrics import (
    apply_output_mask,
    calculate_mae,
    calculate_max_mae,
    calculate_psnr,
    calculate_ssim,
    normalize_to_01,
    reverse_norm,
)


def test_calculate_psnr():
    img1 = torch.randn(1, 1, 32, 32)
    img2 = img1 + torch.randn(1, 1, 32, 32) * 0.1

    psnr = calculate_psnr(img1, img2)

    assert isinstance(psnr, float)
    assert psnr > 0


def test_calculate_psnr_identical():
    img = torch.randn(1, 1, 32, 32)

    psnr = calculate_psnr(img, img)

    assert psnr == float("inf")


def test_calculate_ssim():
    img1 = torch.randn(1, 1, 32, 32)
    img2 = img1 + torch.randn(1, 1, 32, 32) * 0.1

    ssim = calculate_ssim(img1, img2)

    assert isinstance(ssim, float)
    assert -1 <= ssim <= 1


def test_ssim_without_mask_uses_same_weighted_normalization_idea():
    img1 = torch.ones(1, 1, 3, 3)
    img2 = img1.clone()

    ssim = calculate_ssim(img1, img2, kernel_size=3, sigma=1.0)

    assert pytest.approx(ssim, rel=1e-6, abs=1e-6) == 1.0


def test_calculate_mae():
    img1 = torch.randn(1, 1, 32, 32)
    img2 = img1 + torch.randn(1, 1, 32, 32) * 0.1

    mae = calculate_mae(img1, img2)

    assert isinstance(mae, float)
    assert mae >= 0


def test_calculate_max_mae():
    img1 = torch.randn(1, 1, 32, 32)
    img2 = img1 + torch.randn(1, 1, 32, 32) * 0.1

    max_mae = calculate_max_mae(img1, img2)

    assert isinstance(max_mae, float)
    assert max_mae >= 0


def test_reverse_norm():
    img = torch.randn(1, 1, 32, 32)
    mean = torch.tensor([0.5]).view(-1, 1, 1)
    std = torch.tensor([2.0]).view(-1, 1, 1)

    normalized = (img - mean) / std
    recovered = reverse_norm(normalized, mean, std)

    assert torch.allclose(recovered, img, atol=1e-6)


def test_normalize_to_01():
    img1 = torch.randn(1, 1, 32, 32) * 10
    img2 = torch.randn(1, 1, 32, 32) * 10

    norm1, norm2 = normalize_to_01(img1, img2)

    assert norm1.min() >= 0
    assert norm1.max() <= 1
    assert norm2.min() >= 0
    assert norm2.max() <= 1


def test_normalize_to_01_maps_identical_constant_inputs_into_unit_interval():
    img = torch.full((1, 1, 4, 4), 5.0)

    norm1, norm2 = normalize_to_01(img, img)

    assert torch.all(norm1 == 0)
    assert torch.all(norm2 == 0)
    assert pytest.approx(calculate_ssim(norm1, norm2), rel=1e-6, abs=1e-6) == 1.0


def test_metrics_support_multi_channel_inputs_without_special_cases():
    img = torch.randn(2, 2, 16, 16)

    psnr = calculate_psnr(img, img)
    ssim = calculate_ssim(img, img)
    mae = calculate_mae(img, img)
    max_mae = calculate_max_mae(img, img)

    assert psnr == float("inf")
    assert pytest.approx(ssim, rel=1e-6, abs=1e-6) == 1.0
    assert mae == 0.0
    assert max_mae == 0.0


def test_normalize_to_01_supports_multi_channel_inputs():
    hr = torch.randn(1, 2, 8, 8)
    sr = hr + 0.05

    norm_hr, norm_sr = normalize_to_01(hr, sr)

    assert norm_hr.shape == hr.shape
    assert norm_sr.shape == sr.shape


def test_metrics_support_mask_without_channel_specific_logic():
    img = torch.randn(1, 2, 10, 10)
    mask = torch.randint(0, 2, (10, 10), dtype=torch.bool)

    assert calculate_psnr(img, img, mask=mask) == float("inf")
    assert pytest.approx(calculate_ssim(img, img, mask=mask), rel=1e-6, abs=1e-6) == 1.0
    assert calculate_mae(img, img, mask=mask) == 0.0
    assert calculate_max_mae(img, img, mask=mask) == 0.0


def test_masked_ssim_ignores_masked_out_region_in_statistics():
    img1 = torch.zeros(1, 1, 3, 3)
    img2 = img1.clone()
    img1[:, :, 1, 1] = 1.0
    img2[:, :, 1, 1] = 1.0
    img2[:, :, 1, 2] = 0.8
    mask = torch.zeros(3, 3, dtype=torch.bool)
    mask[1, 1] = True

    ssim = calculate_ssim(img1, img2, mask=mask, kernel_size=3, sigma=1.0)

    assert pytest.approx(ssim, rel=1e-6, abs=1e-6) == 1.0


def test_apply_output_mask_broadcasts_over_channels():
    sr = torch.ones(1, 2, 4, 4)
    mask = torch.tensor(
        [
            [True, False, True, False],
            [True, True, False, False],
            [False, False, True, True],
            [True, False, True, False],
        ],
        dtype=torch.bool,
    )

    masked = apply_output_mask(sr, mask)

    assert masked.shape == sr.shape
    assert masked[:, :, 0, 1].sum().item() == 0.0
