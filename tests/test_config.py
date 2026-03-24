"""测试配置系统。"""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from src.app.checkpoint_validation import validate_checkpoint_matches_config
from src.app.config_parsing import load_evaluate_config, load_infer_config, load_train_config
from src.app.config_types import EvaluationLoaderSpec, InferenceLoaderSpec
from src.utils.path import PROJECT_ROOT, resolve_project_path

CONFIG_DIR = str((Path(__file__).resolve().parents[1] / "configs").resolve())


def _compose(config_name: str):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name=config_name)


def test_load_train_config():
    cfg = load_train_config(_compose("wind/train_x2"))

    assert cfg.models.name == "edsr"
    assert cfg.models.in_dim == 1
    assert cfg.upscale == 2
    assert cfg.data_norm.mean == [4.4473377934895]
    assert cfg.data_norm.std == [3.0444616107152527]
    assert cfg.dataset.lr_patch_size == 30
    assert cfg.dataset.train_lr_root
    assert cfg.dataset.val_lr_root
    assert cfg.train.epochs == 100
    assert cfg.train.batch_size == 16
    assert cfg.train.lr == 2e-4
    assert cfg.resume.load_callbacks is True


def test_load_train_config_rejects_bicubic():
    raw_cfg = _compose("wind/train_x2")
    raw_cfg.models.name = "Bicubic"

    with pytest.raises(ValueError, match="bicubic 只支持评估和推理，不支持训练"):
        load_train_config(raw_cfg)


def test_train_models_build_spec():
    cfg = load_train_config(_compose("wind/train_x2"))
    spec = cfg.models_build_spec

    assert spec.model_name == "edsr"
    assert spec.in_dim == 1
    assert spec.upscale == 2


def test_train_loader_spec():
    cfg = load_train_config(_compose("wind/train_x2"))
    spec = cfg.train_loader_spec

    assert spec.lr_patch_size == 30
    assert spec.train_pair.hr_root
    assert spec.val_pair.hr_root
    assert spec.val_pair.max_sample == 10


def test_checkpoint_root():
    cfg = load_train_config(_compose("wind/train_x2"))
    path = cfg.checkpoint_root

    assert "edsr" in str(path)
    assert "x2" in str(path)


def test_load_evaluate_config():
    cfg = load_evaluate_config(_compose("wind/evaluate_x2"))

    assert cfg.models.name == "edsr"
    assert cfg.models.in_dim is None
    assert cfg.upscale == 2
    assert cfg.data_norm.mean == [4.4473377934895]
    assert cfg.dataset.eval_lr_root
    assert cfg.dataset.max_sample is False
    assert cfg.evaluate.batch_size == 1
    assert cfg.mode == "evaluation"
    assert isinstance(cfg.test_loader_spec, EvaluationLoaderSpec)
    assert cfg.test_loader_spec.eval_pair.hr_root


def test_load_evaluate_config_allows_bicubic_without_checkpoint():
    raw_cfg = _compose("wind/evaluate_x2")
    raw_cfg.models.name = "Bicubic"
    raw_cfg.evaluate.checkpoint = None

    cfg = load_evaluate_config(raw_cfg)
    assert cfg.models.name == "bicubic"
    assert cfg.checkpoint is None


def test_load_infer_config():
    cfg = load_infer_config(_compose("wind/infer_x2"))

    assert cfg.models.name == "edsr"
    assert cfg.models.in_dim is None
    assert cfg.upscale == 2
    assert cfg.data_norm.mean == [4.4473377934895]
    assert cfg.dataset.infer_lr_root
    assert cfg.infer.batch_size == 1
    assert cfg.mode == "inference"
    assert isinstance(cfg.test_loader_spec, InferenceLoaderSpec)
    assert cfg.test_loader_spec.infer_lr_root


def test_load_infer_config_allows_bicubic_without_checkpoint():
    raw_cfg = _compose("wind/infer_x2")
    raw_cfg.models.name = "Bicubic"
    raw_cfg.infer.checkpoint = None

    cfg = load_infer_config(raw_cfg)
    assert cfg.models.name == "bicubic"
    assert cfg.checkpoint is None


def test_load_dataset_specific_train_config():
    cfg = load_train_config(_compose("mwd/train_x4"))

    assert cfg.models.name == "edsr"
    assert cfg.models.in_dim == 2
    assert cfg.upscale == 4
    assert cfg.data_norm.mean == [0.4989248121185863, 0.24549368276091751]
    assert cfg.dataset.lr_patch_size == 15
    assert cfg.dataset.train_lr_root.endswith("/workspace/dataset/train/mwd/x4/lr")
    assert cfg.resume.load_callbacks is True


def test_load_dataset_specific_evaluate_config():
    cfg = load_evaluate_config(_compose("wind/evaluate_x2"))

    assert cfg.models.name == "edsr"
    assert cfg.models.in_dim is None
    assert cfg.upscale == 2
    assert cfg.data_norm.std == [3.0444616107152527]
    assert cfg.dataset.eval_lr_root.endswith("/workspace/dataset/test/wind/x2/lr")


def test_load_dataset_specific_infer_config():
    cfg = load_infer_config(_compose("swh/infer_x4"))

    assert cfg.models.name == "edsr"
    assert cfg.models.in_dim is None
    assert cfg.upscale == 4
    assert cfg.data_norm.mean == [0.7569751076884808]
    assert cfg.dataset.infer_lr_root.endswith("/workspace/dataset/test/swh/x4/lr")


def test_load_evaluate_config_accepts_batch_size_gt_one():
    raw_cfg = _compose("wind/evaluate_x2")
    raw_cfg.evaluate.batch_size = 2

    cfg = load_evaluate_config(raw_cfg)
    assert cfg.evaluate.batch_size == 2


def test_load_infer_config_accepts_batch_size_gt_one():
    raw_cfg = _compose("wind/infer_x2")
    raw_cfg.infer.batch_size = 2

    cfg = load_infer_config(raw_cfg)
    assert cfg.infer.batch_size == 2


def test_checkpoint_validation_accepts_legacy_uppercase_model_name(tmp_path):
    cfg = load_train_config(_compose("wind/train_x2"))
    checkpoint_path = tmp_path / "legacy_uppercase_model_name.pth"
    torch.save(
        {
            "config": {
                "models": {
                    "name": "EDSR",
                    "in_dim": cfg.models.in_dim,
                    "params": dict(cfg.models.params),
                },
                "dataset": {
                    "name": cfg.dataset.name,
                },
                "upscale": cfg.upscale,
            }
        },
        checkpoint_path,
    )

    validate_checkpoint_matches_config(cfg, checkpoint_path)


def test_resolve_project_path_resolves_relative_paths_from_repo_root():
    resolved = resolve_project_path("configs/wind/train_x2.yaml")

    assert resolved == PROJECT_ROOT / "configs/wind/train_x2.yaml"
    assert resolved.is_absolute()
