"""数据文件枚举与配对工具。"""

from pathlib import Path


def list_npy_files(root: str | Path) -> list[Path]:
    path = Path(root)
    return sorted(file for file in path.iterdir() if file.is_file() and file.suffix.lower() == ".npy")


def build_paired_npy_files(lr_root: str | Path, hr_root: str | Path) -> list[tuple[Path, Path]]:
    lr_files = {file.name: file for file in list_npy_files(lr_root)}
    hr_files = {file.name: file for file in list_npy_files(hr_root)}

    common_filenames = sorted(set(lr_files) & set(hr_files))
    if not common_filenames:
        raise ValueError(f"LR/HR 路径下没有同名样本: lr_root={lr_root}, hr_root={hr_root}")

    if len(common_filenames) != len(lr_files) or len(common_filenames) != len(hr_files):
        missing_in_lr = sorted(set(hr_files) - set(lr_files))
        missing_in_hr = sorted(set(lr_files) - set(hr_files))
        raise ValueError(
            "LR/HR 文件名不一致。"
            f" 缺少 LR: {missing_in_lr[:5]}"
            f" 缺少 HR: {missing_in_hr[:5]}"
        )

    return [(lr_files[name], hr_files[name]) for name in common_filenames]
