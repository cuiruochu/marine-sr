"""运行前数据校验测试。"""

from pathlib import Path

import numpy as np
import pytest

from src.app.config import load_evaluate_config, load_infer_config, load_train_config
from src.data.validation import validate_eval_runtime_inputs, validate_train_runtime_inputs


def make_train_raw_cfg(train_lr_root: str, train_hr_root: str, val_lr_root: str, val_hr_root: str):
    return {
        "seed": 0,
        "models": {"name": "EDSR", "params": {}},
        "dataset": {
            "param": "wind",
            "upscale": 2,
            "lr_patch_size": 16,
            "channels": 1,
            "normalize": {"mean": [0.0], "std": [1.0]},
            "train_lr_root": train_lr_root,
            "train_hr_root": train_hr_root,
            "val_lr_root": val_lr_root,
            "val_hr_root": val_hr_root,
        },
        "train": {
            "epochs": 2,
            "batch_size": 2,
            "num_workers": 0,
            "val_num_workers": 0,
            "lr": 1e-4,
            "optimizer": {"name": "adam", "params": {}},
            "scheduler": {"name": "none", "params": {}},
            "loss": {"name": "l1", "params": {}},
            "checkpoint": {"every": 1, "save_best": True, "monitor": "val_mae", "mode": "min"},
        },
        "resume": {"checkpoint": None, "load_optimizer": True, "load_scheduler": True, "load_rng_state": True},
        "paths": {"checkpoint_dir": "./checkpoints", "experiment_dir": "./experiments"},
        "wandb": {"project": "marine-sr", "mode": "disabled"},
    }


def make_evaluate_raw_cfg(eval_lr_root: str, eval_hr_root: str):
    return {
        "models": {"name": "EDSR", "params": {}},
        "dataset": {
            "param": "wind",
            "upscale": 2,
            "channels": 1,
            "normalize": {"mean": [0.0], "std": [1.0]},
            "eval_lr_root": eval_lr_root,
            "eval_hr_root": eval_hr_root,
        },
        "evaluate": {
            "checkpoint": "./checkpoints/best.pth",
            "save_results": False,
            "save_dir": "./results",
            "eval_mask": None,
            "batch_size": 1,
            "num_workers": 0,
        },
    }


def make_infer_raw_cfg(infer_lr_root: str):
    return {
        "models": {"name": "EDSR", "params": {}},
        "dataset": {
            "param": "wind",
            "upscale": 2,
            "channels": 1,
            "normalize": {"mean": [0.0], "std": [1.0]},
            "infer_lr_root": infer_lr_root,
        },
        "infer": {
            "checkpoint": "./checkpoints/best.pth",
            "save_results": False,
            "save_dir": "./results",
            "batch_size": 1,
            "num_workers": 0,
        },
    }


def write_npy(root: Path, filename: str):
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / filename, np.zeros((8, 8), dtype=np.float32))


def test_validate_train_runtime_inputs_accepts_paired_train_and_val_dirs(tmp_path: Path):
    train_lr = tmp_path / "train_lr"
    train_hr = tmp_path / "train_hr"
    val_lr = tmp_path / "val_lr"
    val_hr = tmp_path / "val_hr"
    for root in [train_lr, train_hr, val_lr, val_hr]:
        write_npy(root, "sample_001.npy")

    cfg = load_train_config(
        make_train_raw_cfg(str(train_lr), str(train_hr), str(val_lr), str(val_hr))
    )

    validate_train_runtime_inputs(cfg)


def test_validate_train_runtime_inputs_rejects_missing_directory(tmp_path: Path):
    train_lr = tmp_path / "train_lr"
    train_hr = tmp_path / "train_hr"
    val_lr = tmp_path / "val_lr"
    val_hr = tmp_path / "val_hr"
    for root in [train_lr, train_hr, val_lr]:
        write_npy(root, "sample_001.npy")

    cfg = load_train_config(
        make_train_raw_cfg(str(train_lr), str(train_hr), str(val_lr), str(val_hr))
    )

    with pytest.raises(FileNotFoundError, match="dataset.val_hr_root"):
        validate_train_runtime_inputs(cfg)


def test_validate_train_runtime_inputs_rejects_mismatched_pairs(tmp_path: Path):
    train_lr = tmp_path / "train_lr"
    train_hr = tmp_path / "train_hr"
    val_lr = tmp_path / "val_lr"
    val_hr = tmp_path / "val_hr"

    write_npy(train_lr, "sample_001.npy")
    write_npy(train_hr, "sample_002.npy")
    write_npy(val_lr, "sample_001.npy")
    write_npy(val_hr, "sample_001.npy")

    cfg = load_train_config(
        make_train_raw_cfg(str(train_lr), str(train_hr), str(val_lr), str(val_hr))
    )

    with pytest.raises(ValueError, match="LR/HR 路径下没有同名样本|LR/HR 文件名不一致"):
        validate_train_runtime_inputs(cfg)


def test_validate_eval_runtime_inputs_accepts_paired_eval_dirs(tmp_path: Path):
    eval_lr = tmp_path / "eval_lr"
    eval_hr = tmp_path / "eval_hr"
    write_npy(eval_lr, "sample_001.npy")
    write_npy(eval_hr, "sample_001.npy")

    cfg = load_evaluate_config(make_evaluate_raw_cfg(str(eval_lr), str(eval_hr)))

    validate_eval_runtime_inputs(cfg)


def test_validate_eval_runtime_inputs_rejects_empty_infer_dir(tmp_path: Path):
    infer_lr = tmp_path / "infer_lr"
    infer_lr.mkdir(parents=True, exist_ok=True)

    cfg = load_infer_config(make_infer_raw_cfg(str(infer_lr)))

    with pytest.raises(ValueError, match="dataset.infer_lr_root"):
        validate_eval_runtime_inputs(cfg)
