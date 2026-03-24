"""
将 MWD 角度场 `.npy` 文件编码为 cos/sin 双通道表示。

输入要求：
- 单个样本为 `H x W`、`1 x H x W` 或 `H x W x 1`
- 数值默认为角度制（0 到 360）

输出格式：
- `float32`
- `2 x H x W`
- 第 0 通道为 `cos(theta)`，第 1 通道为 `sin(theta)`
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from src.utils.data_files import list_npy_files

LOGGER = logging.getLogger("encode_mwd_cos_sin")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 MWD 角度 `.npy` 文件编码为 cos/sin 双通道表示。")
    parser.add_argument("--input-dir", type=Path, required=True, help="输入 `.npy` 目录。")
    parser.add_argument("--output-dir", type=Path, required=True, help="输出 `.npy` 目录。")
    parser.add_argument(
        "--angle-unit",
        choices=["degrees", "radians"],
        default="degrees",
        help="输入角度单位，默认 degrees。",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")


def main() -> int:
    configure_logging()
    args = parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"输入路径不是目录: {input_dir}")

    resolved_input_dir = input_dir.resolve()
    resolved_output_dir = output_dir.resolve()
    if resolved_input_dir == resolved_output_dir:
        raise ValueError("输入目录与输出目录不能相同")

    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = list_npy_files(input_dir)
    if not input_files:
        raise ValueError(f"目录下没有 `.npy` 文件: {input_dir}")

    for file_path in input_files:
        raw = np.load(file_path, allow_pickle=True)
        angle_hw = to_hw_float32(raw)
        encoded = encode_angle_to_cos_sin(angle_hw, angle_unit=args.angle_unit)
        np.save(output_dir / file_path.name, encoded)

    LOGGER.info("编码完成")
    LOGGER.info("  样本数: %s", len(input_files))
    LOGGER.info("  输出目录: %s", output_dir)
    LOGGER.info("  输出格式: (2, H, W) float32")
    return 0


def to_hw_float32(array: np.ndarray) -> np.ndarray:
    array = fill_none_with_zero(array)
    array = np.nan_to_num(array.astype(np.float32, copy=False), nan=0.0)

    if array.ndim == 2:
        return array

    if array.ndim != 3:
        raise ValueError(f"仅支持二维或单通道三维数组，当前形状={array.shape}")

    if array.shape[0] == 1:
        return array[0]
    if array.shape[2] == 1:
        return array[:, :, 0]

    raise ValueError(f"仅支持单通道角度输入，当前形状={array.shape}")


def fill_none_with_zero(array: np.ndarray) -> np.ndarray:
    if array.dtype != object:
        return array
    return np.vectorize(lambda value: 0.0 if value is None else value, otypes=[object])(array)


def encode_angle_to_cos_sin(angle_hw: np.ndarray, *, angle_unit: str = "degrees") -> np.ndarray:
    if angle_unit == "degrees":
        angle = np.deg2rad(angle_hw)
    elif angle_unit == "radians":
        angle = angle_hw
    else:
        raise ValueError(f"不支持的角度单位: {angle_unit}")

    cos_channel = np.cos(angle).astype(np.float32, copy=False)
    sin_channel = np.sin(angle).astype(np.float32, copy=False)
    return np.stack([cos_channel, sin_channel], axis=0).astype(np.float32, copy=False)


if __name__ == "__main__":
    raise SystemExit(main())
