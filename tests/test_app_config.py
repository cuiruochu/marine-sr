"""任务配置单元测试。"""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.app.config import (
    load_eval_task_config,
    load_evaluate_config,
    load_infer_config,
    load_train_config,
)
from src.utils import PROJECT_ROOT


def make_train_raw_cfg():
    return OmegaConf.create(
        {
            "seed": 7,
            "models": {"name": "RCAN", "params": {"n_feats": 64}},
            "dataset": {
                "name": "mwd",
                "upscale": 4,
                "lr_patch_size": 24,
                "channels": 2,
                "normalize": {"mean": [0.1, 0.2], "std": [0.3, 0.4]},
                "train_lr_root": "./custom_data/mwd/Train/LR/mwd",
                "train_hr_root": "./custom_data/mwd/Train/mwd",
                "val_lr_root": "./custom_data/mwd/Val/LR/mwd",
                "val_hr_root": "./custom_data/mwd/Val/mwd",
                "max_sample": False,
            },
            "train": {
                "epochs": 10,
                "batch_size": 8,
                "num_workers": 2,
                "val_num_workers": 1,
                "lr": 1e-3,
                "optimizer": {"name": "adamw", "params": {"weight_decay": 0.01}},
                "scheduler": {"name": "cosine", "params": {"T_max": 10}},
                "loss": {"name": "l2", "params": {}},
                "checkpoint": {
                    "every": 2,
                    "save_best": True,
                    "monitor": "val_mae",
                    "mode": "min",
                },
            },
            "paths": {"checkpoint_dir": "./checkpoints", "experiment_dir": "./experiments"},
            "wandb": {"project": "marine-sr", "mode": "offline"},
        }
    )


def make_evaluate_raw_cfg():
    return OmegaConf.create(
        {
            "models": {"name": "RCAN", "params": {"n_feats": 64}},
            "dataset": {
                "name": "mwd",
                "upscale": 4,
                "channels": 2,
                "normalize": {"mean": [0.1, 0.2], "std": [0.3, 0.4]},
                "eval_lr_root": "./custom_data/mwd/Test/LR/mwd",
                "eval_hr_root": "./custom_data/mwd/Test/mwd",
                "max_sample": False,
            },
            "evaluate": {
                "checkpoint": "./checkpoints/best.pth",
                "save_results": True,
                "save_dir": "./results",
                "eval_mask": "./mask.npy",
                "batch_size": 1,
                "num_workers": 2,
            },
        }
    )


def make_infer_raw_cfg():
    return OmegaConf.create(
        {
            "models": {"name": "RCAN", "params": {"n_feats": 64}},
            "dataset": {
                "name": "mwd",
                "upscale": 4,
                "channels": 2,
                "normalize": {"mean": [0.1, 0.2], "std": [0.3, 0.4]},
                "infer_lr_root": "./custom_data/mwd/Infer/LR/mwd",
            },
            "infer": {
                "checkpoint": "./checkpoints/best.pth",
                "save_results": True,
                "save_dir": "./results",
                "batch_size": 1,
                "num_workers": 2,
            },
        }
    )


def test_load_train_config_parses_sections():
    cfg = load_train_config(make_train_raw_cfg())

    assert cfg.seed == 7
    assert cfg.model.name == "RCAN"
    assert cfg.dataset.name == "mwd"
    assert cfg.dataset.channels == 2
    assert cfg.dataset.max_sample is False
    assert cfg.train.optimizer.name == "adamw"
    assert cfg.train.num_workers == 2
    assert cfg.train.val_num_workers == 1
    assert cfg.resume.checkpoint is None
    assert cfg.resume.load_optimizer is True
    assert cfg.resume.load_scheduler is True
    assert cfg.resume.load_rng_state is True
    assert cfg.wandb.mode == "offline"


def test_train_app_config_build_specs_and_paths():
    cfg = load_train_config(make_train_raw_cfg())

    assert cfg.model_build_spec.model_name == "RCAN"
    assert cfg.model_build_spec.in_dim == 2
    assert Path(cfg.train_loader_spec.train_lr_root) == PROJECT_ROOT / "custom_data" / "mwd" / "Train" / "LR" / "mwd"
    assert Path(cfg.train_loader_spec.train_hr_root) == PROJECT_ROOT / "custom_data" / "mwd" / "Train" / "mwd"
    assert Path(cfg.train_loader_spec.val_lr_root) == PROJECT_ROOT / "custom_data" / "mwd" / "Val" / "LR" / "mwd"
    assert Path(cfg.train_loader_spec.val_hr_root) == PROJECT_ROOT / "custom_data" / "mwd" / "Val" / "mwd"
    assert cfg.checkpoint_root == PROJECT_ROOT / "checkpoints" / "RCAN" / "mwd" / "x4"


def test_load_evaluate_config_parses_sections_and_paths():
    cfg = load_evaluate_config(make_evaluate_raw_cfg())

    assert cfg.model.name == "RCAN"
    assert cfg.mode == "evaluation"
    assert cfg.evaluate.save_results is True
    assert cfg.test_loader_spec.mode == "evaluation"
    assert Path(cfg.test_loader_spec.lr_root) == PROJECT_ROOT / "custom_data" / "mwd" / "Test" / "LR" / "mwd"
    assert Path(cfg.test_loader_spec.hr_root) == PROJECT_ROOT / "custom_data" / "mwd" / "Test" / "mwd"
    assert cfg.dataset.max_sample is False
    assert cfg.results_root == Path("results") / "RCAN" / "mwd" / "x4"


def test_load_infer_config_parses_sections_and_paths():
    cfg = load_infer_config(make_infer_raw_cfg())

    assert cfg.model.name == "RCAN"
    assert cfg.mode == "inference"
    assert cfg.test_loader_spec.mode == "inference"
    assert Path(cfg.test_loader_spec.lr_root) == PROJECT_ROOT / "custom_data" / "mwd" / "Infer" / "LR" / "mwd"
    assert cfg.test_loader_spec.hr_root is None
    assert cfg.results_root == Path("results") / "RCAN" / "mwd" / "x4"


def test_load_train_config_rejects_invalid_model_name():
    raw_cfg = make_train_raw_cfg()
    raw_cfg.models.name = "InvalidModel"

    with pytest.raises(ValueError, match="model.name"):
        load_train_config(raw_cfg)


def test_load_train_config_rejects_invalid_optimizer_name():
    raw_cfg = make_train_raw_cfg()
    raw_cfg.train.optimizer.name = "invalid_optimizer"

    with pytest.raises(ValueError, match="train.optimizer.name"):
        load_train_config(raw_cfg)


def test_load_train_config_rejects_invalid_scheduler_name():
    raw_cfg = make_train_raw_cfg()
    raw_cfg.train.scheduler.name = "invalid_scheduler"

    with pytest.raises(ValueError, match="train.scheduler.name"):
        load_train_config(raw_cfg)


def test_load_train_config_allows_scheduler_none():
    raw_cfg = make_train_raw_cfg()
    raw_cfg.train.scheduler.name = "none"

    cfg = load_train_config(raw_cfg)
    assert cfg.train.scheduler.name == "none"


def test_load_train_config_rejects_invalid_loss_name():
    raw_cfg = make_train_raw_cfg()
    raw_cfg.train.loss.name = "invalid_loss"

    with pytest.raises(ValueError, match="train.loss.name"):
        load_train_config(raw_cfg)


def test_load_train_config_rejects_invalid_normalize_length():
    raw_cfg = make_train_raw_cfg()
    raw_cfg.dataset.normalize.mean = [0.1]

    with pytest.raises(ValueError, match="dataset.normalize.mean"):
        load_train_config(raw_cfg)


def test_load_infer_config_rejects_non_positive_std():
    raw_cfg = make_infer_raw_cfg()
    raw_cfg.dataset.normalize.std = [0.3, 0.0]

    with pytest.raises(ValueError, match="dataset.normalize.std"):
        load_infer_config(raw_cfg)


def test_load_train_config_rejects_non_positive_batch_size():
    raw_cfg = make_train_raw_cfg()
    raw_cfg.train.batch_size = 0

    with pytest.raises(ValueError, match="train.batch_size"):
        load_train_config(raw_cfg)


def test_load_evaluate_config_rejects_non_positive_batch_size():
    raw_cfg = make_evaluate_raw_cfg()
    raw_cfg.evaluate.batch_size = 0

    with pytest.raises(ValueError, match="evaluate.batch_size"):
        load_evaluate_config(raw_cfg)


def test_load_infer_config_rejects_non_positive_batch_size():
    raw_cfg = make_infer_raw_cfg()
    raw_cfg.infer.batch_size = 0

    with pytest.raises(ValueError, match="infer.batch_size"):
        load_infer_config(raw_cfg)


def test_load_train_config_rejects_invalid_wandb_mode():
    raw_cfg = make_train_raw_cfg()
    raw_cfg.wandb.mode = "local"

    with pytest.raises(ValueError, match="wandb.mode"):
        load_train_config(raw_cfg)


def test_load_evaluate_config_rejects_blank_checkpoint():
    raw_cfg = make_evaluate_raw_cfg()
    raw_cfg.evaluate.checkpoint = "   "

    with pytest.raises(ValueError, match="evaluate.checkpoint"):
        load_evaluate_config(raw_cfg)


def test_load_infer_config_rejects_blank_checkpoint():
    raw_cfg = make_infer_raw_cfg()
    raw_cfg.infer.checkpoint = "   "

    with pytest.raises(ValueError, match="infer.checkpoint"):
        load_infer_config(raw_cfg)


def test_load_train_config_rejects_blank_dataset_train_path():
    raw_cfg = make_train_raw_cfg()
    raw_cfg.dataset.train_lr_root = "   "

    with pytest.raises(ValueError, match="dataset.train_lr_root"):
        load_train_config(raw_cfg)


def test_load_train_config_parses_resume_section():
    raw_cfg = make_train_raw_cfg()
    raw_cfg.resume = {
        "checkpoint": "./checkpoints/last.pth",
        "load_optimizer": False,
        "load_scheduler": False,
        "load_rng_state": False,
    }

    cfg = load_train_config(raw_cfg)

    assert cfg.resume.checkpoint == "./checkpoints/last.pth"
    assert cfg.resume.load_optimizer is False
    assert cfg.resume.load_scheduler is False
    assert cfg.resume.load_rng_state is False


def test_load_train_config_rejects_blank_resume_checkpoint():
    raw_cfg = make_train_raw_cfg()
    raw_cfg.resume = {
        "checkpoint": "   ",
        "load_optimizer": True,
        "load_scheduler": True,
        "load_rng_state": True,
    }

    with pytest.raises(ValueError, match="resume.checkpoint"):
        load_train_config(raw_cfg)


def test_load_train_config_rejects_resume_checkpoint_without_supported_suffix():
    raw_cfg = make_train_raw_cfg()
    raw_cfg.resume = {
        "checkpoint": "./checkpoints/last.bin",
        "load_optimizer": True,
        "load_scheduler": True,
        "load_rng_state": True,
    }

    with pytest.raises(ValueError, match="resume.checkpoint"):
        load_train_config(raw_cfg)


def test_load_evaluate_config_rejects_blank_dataset_eval_path():
    raw_cfg = make_evaluate_raw_cfg()
    raw_cfg.dataset.eval_hr_root = "   "

    with pytest.raises(ValueError, match="dataset.eval_hr_root"):
        load_evaluate_config(raw_cfg)


def test_load_infer_config_rejects_blank_dataset_infer_path():
    raw_cfg = make_infer_raw_cfg()
    raw_cfg.dataset.infer_lr_root = "   "

    with pytest.raises(ValueError, match="dataset.infer_lr_root"):
        load_infer_config(raw_cfg)


def test_load_train_config_rejects_same_train_lr_hr_root():
    raw_cfg = make_train_raw_cfg()
    raw_cfg.dataset.train_hr_root = raw_cfg.dataset.train_lr_root

    with pytest.raises(ValueError, match="dataset.train_lr_root"):
        load_train_config(raw_cfg)


def test_load_evaluate_config_rejects_same_eval_lr_hr_root():
    raw_cfg = make_evaluate_raw_cfg()
    raw_cfg.dataset.eval_hr_root = raw_cfg.dataset.eval_lr_root

    with pytest.raises(ValueError, match="dataset.eval_lr_root"):
        load_evaluate_config(raw_cfg)


def test_load_train_config_rejects_checkpoint_every_greater_than_epochs():
    raw_cfg = make_train_raw_cfg()
    raw_cfg.train.checkpoint.every = 11

    with pytest.raises(ValueError, match="train.checkpoint.every"):
        load_train_config(raw_cfg)


def test_load_evaluate_config_rejects_checkpoint_without_supported_suffix():
    raw_cfg = make_evaluate_raw_cfg()
    raw_cfg.evaluate.checkpoint = "./checkpoints/best.ckpt"

    with pytest.raises(ValueError, match="evaluate.checkpoint"):
        load_evaluate_config(raw_cfg)


def test_load_infer_config_rejects_checkpoint_without_supported_suffix():
    raw_cfg = make_infer_raw_cfg()
    raw_cfg.infer.checkpoint = "./checkpoints/best.ckpt"

    with pytest.raises(ValueError, match="infer.checkpoint"):
        load_infer_config(raw_cfg)


def test_load_evaluate_config_normalizes_blank_eval_mask_to_none():
    raw_cfg = make_evaluate_raw_cfg()
    raw_cfg.evaluate.eval_mask = "   "

    cfg = load_evaluate_config(raw_cfg)

    assert cfg.evaluate.eval_mask is None


def test_load_evaluate_config_rejects_eval_mask_without_npy_suffix():
    raw_cfg = make_evaluate_raw_cfg()
    raw_cfg.evaluate.eval_mask = "./mask.png"

    with pytest.raises(ValueError, match="evaluate.eval_mask"):
        load_evaluate_config(raw_cfg)


def test_load_eval_task_config_dispatches_evaluate():
    cfg = load_eval_task_config(make_evaluate_raw_cfg())

    assert cfg.mode == "evaluation"


def test_load_eval_task_config_dispatches_infer():
    cfg = load_eval_task_config(make_infer_raw_cfg())

    assert cfg.mode == "inference"
