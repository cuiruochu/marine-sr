"""Runtime input validation for training and evaluation."""

from __future__ import annotations

from src.utils.data_files import build_paired_npy_files, list_npy_files


def validate_train_runtime_inputs(cfg) -> None:
    _ensure_paired_root("dataset.train_lr_root", cfg.dataset.train_lr_root, "dataset.train_hr_root", cfg.dataset.train_hr_root)
    _ensure_paired_root("dataset.val_lr_root", cfg.dataset.val_lr_root, "dataset.val_hr_root", cfg.dataset.val_hr_root)


def validate_eval_runtime_inputs(cfg) -> None:
    if getattr(cfg, "mode", None) == "inference":
        _ensure_single_root("dataset.infer_lr_root", cfg.dataset.infer_lr_root)
        return

    _ensure_paired_root("dataset.eval_lr_root", cfg.dataset.eval_lr_root, "dataset.eval_hr_root", cfg.dataset.eval_hr_root)


def _ensure_single_root(field_name: str, root: str) -> None:
    files = list_npy_files(_ensure_existing_dir(field_name, root))
    if not files:
        raise ValueError(f"{field_name} 下没有可用的 .npy 文件: {root}")


def _ensure_paired_root(lr_field: str, lr_root: str, hr_field: str, hr_root: str) -> None:
    _ensure_existing_dir(lr_field, lr_root)
    _ensure_existing_dir(hr_field, hr_root)
    build_paired_npy_files(lr_root, hr_root)


def _ensure_existing_dir(field_name: str, path: str) -> str:
    from pathlib import Path

    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"{field_name} 不存在: {path}")
    if not root.is_dir():
        raise NotADirectoryError(f"{field_name} 不是目录: {path}")
    return str(root)
