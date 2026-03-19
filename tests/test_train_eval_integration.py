"""最小真实训练、评估、推理集成测试。"""

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf


class DummyLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


def _write_fake_samples(root: Path, count: int, size: int = 16, *, channels: int = 1, angle_mode: bool = False):
    root.mkdir(parents=True, exist_ok=True)
    for idx in range(count):
        if channels == 1:
            sample = np.random.rand(size, size).astype(np.float32)
        else:
            sample = np.random.rand(channels, size, size).astype(np.float32)
        if angle_mode:
            sample *= 360.0
        np.save(root / f"sample_{idx}.npy", sample)


def _write_fake_lr_hr_samples(
    root: Path,
    split: str,
    dataset_param: str,
    count: int,
    *,
    channels: int = 1,
    angle_mode: bool = False,
):
    _write_fake_samples(root / split / dataset_param, count=count, size=16, channels=channels, angle_mode=angle_mode)
    _write_fake_samples(root / split / "LR" / dataset_param, count=count, size=8, channels=channels, angle_mode=angle_mode)


def _make_tiny_train_cfg(
    *,
    dataset_param: str,
    channels: int,
    normalize_mean: list[float],
    normalize_std: list[float],
):
    return OmegaConf.create(
        {
            "seed": 0,
            "models": {
                "name": "EDSR",
                "params": {
                    "n_feats": 4,
                    "n_resblocks": 1,
                    "res_scale": 0.1,
                },
            },
            "dataset": {
                "name": dataset_param,
                "upscale": 2,
                "lr_patch_size": 8,
                "channels": channels,
                "normalize": {"mean": normalize_mean, "std": normalize_std},
                "train_lr_root": f"./data/{dataset_param}/Train/LR/{dataset_param}",
                "train_hr_root": f"./data/{dataset_param}/Train/{dataset_param}",
                "val_lr_root": f"./data/{dataset_param}/Val/LR/{dataset_param}",
                "val_hr_root": f"./data/{dataset_param}/Val/{dataset_param}",
                "max_sample": False,
            },
            "train": {
                "epochs": 1,
                "batch_size": 1,
                "num_workers": 0,
                "val_num_workers": 0,
                "lr": 1e-3,
                "optimizer": {"name": "adam", "params": {}},
                "scheduler": {"name": "none", "params": {}},
                "loss": {"name": "l1", "params": {}},
                "checkpoint": {
                    "every": 1,
                    "save_best": True,
                    "monitor": "val_mae",
                    "mode": "min",
                },
            },
            "paths": {
                "checkpoint_dir": "checkpoints",
                "experiment_dir": "experiments",
            },
            "wandb": {
                "project": "marine-sr-test",
                "mode": "disabled",
            },
        }
    )


def _make_tiny_evaluate_cfg(
    tmp_path: Path,
    *,
    dataset_param: str,
    channels: int,
    normalize_mean: list[float],
    normalize_std: list[float],
    checkpoint: str,
    save_results: bool,
):
    return OmegaConf.create(
        {
            "models": {
                "name": "EDSR",
                "params": {
                    "n_feats": 4,
                    "n_resblocks": 1,
                    "res_scale": 0.1,
                },
            },
            "dataset": {
                "name": dataset_param,
                "upscale": 2,
                "channels": channels,
                "normalize": {"mean": normalize_mean, "std": normalize_std},
                "eval_lr_root": f"./data/{dataset_param}/Test/LR/{dataset_param}",
                "eval_hr_root": f"./data/{dataset_param}/Test/{dataset_param}",
                "max_sample": False,
            },
            "evaluate": {
                "checkpoint": checkpoint,
                "save_results": save_results,
                "save_dir": str(tmp_path / "eval_results"),
                "eval_mask": None,
                "batch_size": 1,
                "num_workers": 0,
            },
        }
    )


def _make_tiny_infer_cfg(
    tmp_path: Path,
    *,
    dataset_param: str,
    channels: int,
    normalize_mean: list[float],
    normalize_std: list[float],
    checkpoint: str,
    save_results: bool,
):
    return OmegaConf.create(
        {
            "models": {
                "name": "EDSR",
                "params": {
                    "n_feats": 4,
                    "n_resblocks": 1,
                    "res_scale": 0.1,
                },
            },
            "dataset": {
                "name": dataset_param,
                "upscale": 2,
                "channels": channels,
                "normalize": {"mean": normalize_mean, "std": normalize_std},
                "infer_lr_root": f"./data/{dataset_param}/Infer/LR/{dataset_param}",
            },
            "infer": {
                "checkpoint": checkpoint,
                "save_results": save_results,
                "save_dir": str(tmp_path / "infer_results"),
                "batch_size": 1,
                "num_workers": 0,
            },
        }
    )


@pytest.mark.parametrize(
    ("dataset_param", "channels", "normalize_mean", "normalize_std", "angle_mode"),
    [
        ("wind", 1, [0.0], [1.0], False),
        ("mwd", 2, [0.0, 0.0], [1.0, 1.0], True),
    ],
)
def test_tiny_train_then_eval_and_infer_integration(
    tmp_path,
    monkeypatch,
    dataset_param,
    channels,
    normalize_mean,
    normalize_std,
    angle_mode,
):
    import src.app.config as app_config_module
    import src.app.eval as eval_app
    import src.app.train as train_app
    from src.app.config import load_evaluate_config, load_infer_config, load_train_config

    monkeypatch.setattr(app_config_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(train_app, "init_logger", lambda: None)
    monkeypatch.setattr(train_app, "get_logger", lambda: DummyLogger())
    monkeypatch.setattr(eval_app, "init_logger", lambda: None)
    monkeypatch.setattr(eval_app, "get_logger", lambda: DummyLogger())

    data_root = tmp_path / "data" / dataset_param
    _write_fake_lr_hr_samples(data_root, "Train", dataset_param, count=2, channels=channels, angle_mode=angle_mode)
    _write_fake_lr_hr_samples(data_root, "Val", dataset_param, count=2, channels=channels, angle_mode=angle_mode)
    _write_fake_lr_hr_samples(data_root, "Test", dataset_param, count=2, channels=channels, angle_mode=angle_mode)
    _write_fake_samples(data_root / "Infer" / "LR" / dataset_param, count=2, size=8, channels=channels, angle_mode=angle_mode)

    train_cfg = load_train_config(
        _make_tiny_train_cfg(
            dataset_param=dataset_param,
            channels=channels,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )
    )
    train_app.run_training(train_cfg)

    checkpoint_dir = tmp_path / "checkpoints" / "EDSR" / dataset_param / "x2"
    best_checkpoint = checkpoint_dir / "best.pth"
    last_checkpoint = checkpoint_dir / "last.pth"
    epoch_checkpoint = checkpoint_dir / "epoch_1.pth"

    assert best_checkpoint.exists()
    assert last_checkpoint.exists()
    assert epoch_checkpoint.exists()

    eval_cfg = load_evaluate_config(
        _make_tiny_evaluate_cfg(
            tmp_path,
            dataset_param=dataset_param,
            channels=channels,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
            checkpoint=str(best_checkpoint),
            save_results=True,
        )
    )
    eval_app.run_evaluation(eval_cfg)

    eval_results_dir = tmp_path / "eval_results" / "EDSR" / dataset_param / "x2"
    eval_result_files = sorted(eval_results_dir.glob("*.npy"))

    assert len(eval_result_files) == 2
    eval_result = np.load(eval_result_files[0])
    assert eval_result.shape == (channels, 16, 16)

    infer_cfg = load_infer_config(
        _make_tiny_infer_cfg(
            tmp_path,
            dataset_param=dataset_param,
            channels=channels,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
            checkpoint=str(best_checkpoint),
            save_results=True,
        )
    )
    eval_app.run_evaluation(infer_cfg)

    infer_results_dir = tmp_path / "infer_results" / "EDSR" / dataset_param / "x2"
    infer_result_files = sorted(infer_results_dir.glob("*.npy"))

    assert len(infer_result_files) == 2
    infer_result = np.load(infer_result_files[0])
    assert infer_result.shape == (channels, 16, 16)
