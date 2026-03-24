"""
数据文件工具
"""

from pathlib import Path
from typing import List, Tuple


def list_npy_files(root: str) -> List[Path]:
    """列出目录下所有 .npy 文件。"""
    root_path = Path(root)
    if not root_path.exists():
        return []
    return sorted(root_path.glob("*.npy"))


def build_paired_npy_files(lr_root: str, hr_root: str) -> List[Tuple[Path, Path]]:
    """构建 LR/HR 配对文件列表。"""
    lr_files = list_npy_files(lr_root)
    hr_files = list_npy_files(hr_root)

    hr_names = {f.name: f for f in hr_files}

    pairs = []
    for lr_file in lr_files:
        if lr_file.name in hr_names:
            pairs.append((lr_file, hr_names[lr_file.name]))

    return pairs


def validate_paired_files(lr_root: str, hr_root: str) -> Tuple[bool, str]:
    """验证 LR/HR 文件配对。"""
    lr_files = list_npy_files(lr_root)
    hr_files = list_npy_files(hr_root)

    if not lr_files:
        return False, f"LR 目录为空: {lr_root}"
    if not hr_files:
        return False, f"HR 目录为空: {hr_root}"

    hr_names = {f.name for f in hr_files}
    lr_names = {f.name for f in lr_files}

    missing_in_hr = lr_names - hr_names
    missing_in_lr = hr_names - lr_names

    if missing_in_hr:
        return False, f"HR 目录缺少文件: {missing_in_hr}"
    if missing_in_lr:
        return False, f"LR 目录缺少文件: {missing_in_lr}"

    return True, "OK"
