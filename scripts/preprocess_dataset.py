"""
将一个 HR `.npy` 目录预处理成训练可用的 HR/LR 配对数据。

功能：
1. 读取输入 HR 目录中的所有 `.npy` 文件
2. 将样本统一保存为 `float32` 的 `(C, H, W)`
3. 若 `H/W` 不能被 `scale` 整除，则裁掉右侧和下侧多余像素
4. 使用 torchvision 的 bicubic 插值下采样生成 LR
5. 按原文件名分别保存处理后的 HR 和 LR
6. 统计所有处理后 HR 样本的整体 mean/std，并写入 `stats.json`

说明：
- 原始 HR 输入目录与处理后 HR 输出目录必须不同，避免覆盖原始数据
- 若样本中包含 `None`，默认填成 `0`
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms.functional import InterpolationMode, resize

from src.utils.data_files import list_npy_files


LOGGER = logging.getLogger("preprocess_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="预处理 HR `.npy` 数据，导出标准化的 HR/LR 配对文件。")
    parser.add_argument("--hr-input-dir", type=Path, required=True, help="原始 HR `.npy` 目录。")
    parser.add_argument("--hr-output-dir", type=Path, required=True, help="处理后 HR 输出目录。")
    parser.add_argument("--lr-output-dir", type=Path, required=True, help="下采样 LR 输出目录。")
    parser.add_argument("--scale", type=int, default=2, help="下采样倍率，默认 2。")
    parser.add_argument(
        "--input-layout",
        choices=["auto", "chw", "hwc"],
        default="auto",
        help="三维输入数组的通道布局。默认 auto。",
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=None,
        help="统计结果输出路径。默认写到 hr-output-dir 同级目录下的 stats.json。",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")


def main() -> int:
    configure_logging()
    args = parse_args()

    hr_input_dir = args.hr_input_dir
    hr_output_dir = args.hr_output_dir
    lr_output_dir = args.lr_output_dir
    stats_path = args.stats_path or hr_output_dir.parent / "stats.json"

    if not hr_input_dir.exists():
        raise FileNotFoundError(f"HR 输入目录不存在: {hr_input_dir}")
    if not hr_input_dir.is_dir():
        raise NotADirectoryError(f"HR 输入路径不是目录: {hr_input_dir}")
    if args.scale <= 1:
        raise ValueError("--scale 必须大于 1")

    resolved_hr_input_dir = hr_input_dir.resolve()
    resolved_hr_output_dir = hr_output_dir.resolve()
    resolved_lr_output_dir = lr_output_dir.resolve()

    if resolved_hr_input_dir == resolved_hr_output_dir:
        raise ValueError("原始 HR 输入目录与处理后 HR 输出目录不能相同")
    if resolved_hr_input_dir == resolved_lr_output_dir:
        raise ValueError("原始 HR 输入目录与 LR 输出目录不能相同")
    if resolved_hr_output_dir == resolved_lr_output_dir:
        raise ValueError("处理后 HR 输出目录与 LR 输出目录不能相同")

    hr_output_dir.mkdir(parents=True, exist_ok=True)
    lr_output_dir.mkdir(parents=True, exist_ok=True)

    hr_files = list_npy_files(hr_input_dir)
    if not hr_files:
        raise ValueError(f"目录下没有 `.npy` 文件: {hr_input_dir}")

    channel_count: int | None = None
    channel_sum: np.ndarray | None = None
    channel_sum_sq: np.ndarray | None = None
    pixel_count_per_channel = 0
    trimmed_files: list[str] = []

    for file_path in hr_files:
        raw = np.load(file_path)
        hr = to_chw_float32(raw, layout=args.input_layout)
        hr = trim_to_scale(hr, args.scale)

        if hr.shape[1] == 0 or hr.shape[2] == 0:
            raise ValueError(f"样本尺寸在裁剪后为空: {file_path}, shape={tuple(hr.shape)}")

        lr = bicubic_downsample(hr, args.scale)

        np.save(hr_output_dir / file_path.name, hr)
        np.save(lr_output_dir / file_path.name, lr)

        if channel_count is None:
            channel_count = hr.shape[0]
            channel_sum = np.zeros(channel_count, dtype=np.float64)
            channel_sum_sq = np.zeros(channel_count, dtype=np.float64)
        elif hr.shape[0] != channel_count:
            raise ValueError(
                f"通道数不一致: 期望 {channel_count}，当前文件 {file_path.name} 为 {hr.shape[0]} 通道"
            )

        assert channel_sum is not None
        assert channel_sum_sq is not None
        channel_sum += hr.sum(axis=(1, 2), dtype=np.float64)
        channel_sum_sq += np.square(hr, dtype=np.float64).sum(axis=(1, 2), dtype=np.float64)
        pixel_count_per_channel += hr.shape[1] * hr.shape[2]

        original = infer_original_chw_shape(raw, args.input_layout)
        if original != hr.shape:
            trimmed_files.append(file_path.name)

    assert channel_count is not None
    assert channel_sum is not None
    assert channel_sum_sq is not None

    mean = channel_sum / pixel_count_per_channel
    variance = channel_sum_sq / pixel_count_per_channel - np.square(mean)
    std = np.sqrt(np.maximum(variance, 0.0))

    stats_payload = {
        "hr_input_dir": str(hr_input_dir),
        "hr_output_dir": str(hr_output_dir),
        "lr_output_dir": str(lr_output_dir),
        "scale": args.scale,
        "input_layout": args.input_layout,
        "num_samples": len(hr_files),
        "channels": channel_count,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "trimmed_files": trimmed_files,
    }
    stats_path.write_text(json.dumps(stats_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    LOGGER.info("预处理完成")
    LOGGER.info("  样本数: %s", len(hr_files))
    LOGGER.info("  通道数: %s", channel_count)
    LOGGER.info("  mean: %s", [round(value, 6) for value in mean.tolist()])
    LOGGER.info("  std: %s", [round(value, 6) for value in std.tolist()])
    LOGGER.info("  HR 输出: %s", hr_output_dir)
    LOGGER.info("  LR 输出: %s", lr_output_dir)
    LOGGER.info("  统计文件: %s", stats_path)
    if trimmed_files:
        LOGGER.warning("  有 %s 个文件因不能整除 scale 而被裁剪", len(trimmed_files))

    return 0


def to_chw_float32(array: np.ndarray, layout: str) -> np.ndarray:
    array = fill_none_with_zero(array)

    if array.ndim == 2:
        return np.nan_to_num(array.astype(np.float32, copy=False), nan=0.0)[None, ...]

    if array.ndim != 3:
        raise ValueError(f"仅支持二维(H, W)或三维数组，当前形状={array.shape}")

    if layout == "chw":
        return np.nan_to_num(array.astype(np.float32, copy=False), nan=0.0)

    if layout == "hwc":
        return np.nan_to_num(np.transpose(array, (2, 0, 1)).astype(np.float32, copy=False), nan=0.0)

    if array.shape[0] <= 8 and array.shape[1] > 8 and array.shape[2] > 8:
        return np.nan_to_num(array.astype(np.float32, copy=False), nan=0.0)
    if array.shape[2] <= 8 and array.shape[0] > 8 and array.shape[1] > 8:
        return np.nan_to_num(np.transpose(array, (2, 0, 1)).astype(np.float32, copy=False), nan=0.0)

    raise ValueError(
        "无法在 auto 模式下推断三维数组布局。"
        f" 当前形状={array.shape}。请显式传入 --input-layout chw 或 --input-layout hwc"
    )


def fill_none_with_zero(array: np.ndarray) -> np.ndarray:
    if array.dtype != object:
        return array

    return np.vectorize(lambda value: 0.0 if value is None else value, otypes=[object])(array)


def infer_original_chw_shape(array: np.ndarray, layout: str) -> tuple[int, int, int]:
    chw = to_chw_float32(array, layout)
    return tuple(chw.shape)


def trim_to_scale(chw: np.ndarray, scale: int) -> np.ndarray:
    channels, height, width = chw.shape
    trimmed_height = height - (height % scale)
    trimmed_width = width - (width % scale)
    if trimmed_height == height and trimmed_width == width:
        return chw
    return chw[:, :trimmed_height, :trimmed_width]


def bicubic_downsample(chw: np.ndarray, scale: int) -> np.ndarray:
    tensor = torch.from_numpy(chw)
    _, height, width = tensor.shape
    lr = resize(
        tensor,
        [height // scale, width // scale],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    return lr.to(dtype=torch.float32).numpy()


if __name__ == "__main__":
    raise SystemExit(main())
