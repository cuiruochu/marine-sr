"""运行时数据验证。"""

import logging

import numpy as np

from src.app.config_types import EvaluateAppConfig, InferAppConfig
from src.utils.data_files import list_npy_files, validate_paired_files
from src.utils.path import resolve_project_path

logger = logging.getLogger(__name__)


def validate_train_runtime_inputs(cfg) -> None:
    train_pair = cfg.train_loader_spec.train_pair
    _validate_data_dir(train_pair.lr_root, "dataset.train_pair.lr_root")
    _validate_data_dir(train_pair.hr_root, "dataset.train_pair.hr_root")
    ok, msg = validate_paired_files(train_pair.lr_root, train_pair.hr_root)
    if not ok:
        raise ValueError(f"训练数据配对失败: {msg}")

    val_pair = cfg.train_loader_spec.val_pair
    _validate_data_dir(val_pair.lr_root, "dataset.val_pair.lr_root")
    _validate_data_dir(val_pair.hr_root, "dataset.val_pair.hr_root")
    ok, msg = validate_paired_files(val_pair.lr_root, val_pair.hr_root)
    if not ok:
        raise ValueError(f"验证数据配对失败: {msg}")


def validate_eval_runtime_inputs(cfg) -> None:
    if isinstance(cfg, EvaluateAppConfig):
        validate_evaluation_runtime_inputs(cfg)
        return

    validate_inference_runtime_inputs(cfg)


def validate_evaluation_runtime_inputs(cfg: EvaluateAppConfig) -> None:
    pair = cfg.test_loader_spec.eval_pair
    _validate_data_dir(pair.lr_root, "dataset.eval_pair.lr_root")
    _validate_data_dir(pair.hr_root, "dataset.eval_pair.hr_root")
    ok, msg = validate_paired_files(pair.lr_root, pair.hr_root)
    if not ok:
        raise ValueError(f"评估数据配对失败: {msg}")
    _validate_mask(cfg.mask, pair.hr_root, field_name="evaluate.mask", upscale=None)


def validate_inference_runtime_inputs(cfg: InferAppConfig) -> None:
    _validate_data_dir(cfg.test_loader_spec.infer_lr_root, "dataset.infer_lr_root")
    _validate_mask(cfg.mask, cfg.test_loader_spec.infer_lr_root, field_name="infer.mask", upscale=cfg.upscale)


def _validate_data_dir(path: str, field_name: str) -> None:
    p = resolve_project_path(path)
    if not p.exists():
        raise FileNotFoundError(f"{field_name} 路径不存在: {path}")
    if not p.is_dir():
        raise NotADirectoryError(f"{field_name} 不是目录: {path}")
    if not list_npy_files(str(p)):
        raise ValueError(f"{field_name} 目录下没有 .npy 文件: {path}")


def _validate_mask(mask_path: str | None, sample_root: str, *, field_name: str, upscale: int | None) -> None:
    if not mask_path:
        return

    path = resolve_project_path(mask_path)
    if not path.exists():
        raise FileNotFoundError(f"{field_name} 路径不存在: {mask_path}")
    if path.suffix.lower() != ".npy":
        raise ValueError(f"{field_name} 必须是 .npy 文件")

    mask = np.load(path)
    if mask.dtype != np.bool_:
        raise TypeError(f"{field_name} 必须是 bool 类型，当前 dtype={mask.dtype}")
    if mask.ndim != 2:
        raise ValueError(f"{field_name} shape 必须是 (H, W)，当前={mask.shape}")

    sample_path = list_npy_files(str(resolve_project_path(sample_root)))[0]
    sample = np.load(sample_path, mmap_mode="r")
    if sample.ndim == 2:
        h, w = sample.shape
    elif sample.ndim == 3:
        h, w = sample.shape[-2:]
    else:
        raise ValueError(f"{field_name} 无法推断样本空间尺寸: sample shape={sample.shape}")

    expected_shape = (h, w) if upscale is None else (h * upscale, w * upscale)
    if tuple(mask.shape) != expected_shape:
        raise ValueError(f"{field_name} shape 必须等于 {expected_shape}，当前={tuple(mask.shape)}")
