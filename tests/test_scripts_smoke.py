"""脚本入口与运行时 smoke tests。"""

from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.app.config import load_evaluate_config, load_infer_config, load_train_config


class DummyLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv(x)


def make_train_raw_cfg():
    return OmegaConf.create(
        {
            "seed": 0,
            "models": {"name": "EDSR", "params": {}},
            "dataset": {
                "param": "wind",
                "upscale": 2,
                "lr_patch_size": 30,
                "channels": 1,
                "normalize": {"mean": [0.0], "std": [1.0]},
                "train_lr_root": "./data/wind/Train/LR/wind",
                "train_hr_root": "./data/wind/Train/wind",
                "val_lr_root": "./data/wind/Val/LR/wind",
                "val_hr_root": "./data/wind/Val/wind",
            },
            "train": {
                "epochs": 2,
                "batch_size": 4,
                "num_workers": 0,
                "val_num_workers": 0,
                "lr": 1e-4,
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
            "paths": {"checkpoint_dir": "checkpoints", "experiment_dir": "experiments"},
            "wandb": {"mode": "disabled", "project": "test"},
        }
    )


def make_evaluate_raw_cfg(*, checkpoint=None):
    return OmegaConf.create(
        {
            "models": {"name": "EDSR", "params": {}},
            "dataset": {
                "param": "wind",
                "upscale": 2,
                "channels": 1,
                "normalize": {"mean": [0.0], "std": [1.0]},
                "eval_lr_root": "./data/wind/Test/LR/wind",
                "eval_hr_root": "./data/wind/Test/wind",
            },
            "evaluate": {
                "checkpoint": checkpoint,
                "save_results": False,
                "save_dir": "results",
                "eval_mask": "",
                "batch_size": 1,
                "num_workers": 4,
            },
        }
    )


def make_infer_raw_cfg(*, checkpoint=None):
    return OmegaConf.create(
        {
            "models": {"name": "EDSR", "params": {}},
            "dataset": {
                "param": "wind",
                "upscale": 2,
                "channels": 1,
                "normalize": {"mean": [0.0], "std": [1.0]},
                "infer_lr_root": "./data/wind/Infer/LR/wind",
            },
            "infer": {
                "checkpoint": checkpoint,
                "save_results": True,
                "save_dir": "results",
                "batch_size": 1,
                "num_workers": 4,
            },
        }
    )


def save_checkpoint_with_config(path: Path, config: dict):
    torch.save({"epoch": 1, "config": config, "model": {}}, path)


def test_train_script_main_smoke(monkeypatch):
    import scripts.train as train_script

    calls = {}
    monkeypatch.setattr(train_script, "run_training", lambda cfg: calls.setdefault("cfg", cfg))

    cfg = make_train_raw_cfg()
    train_script.main.__wrapped__(cfg)

    assert calls["cfg"] is cfg


def test_evaluate_script_main_smoke(monkeypatch, tmp_path):
    import scripts.evaluate as evaluate_script

    calls = {}
    cfg = make_evaluate_raw_cfg(checkpoint=str(tmp_path / "best.pth"))
    monkeypatch.setattr(evaluate_script, "run_evaluation", lambda cfg: calls.setdefault("cfg", cfg))

    evaluate_script.main.__wrapped__(cfg)

    assert calls["cfg"] is cfg


def test_infer_script_main_smoke(monkeypatch, tmp_path):
    import scripts.infer as infer_script

    calls = {}
    cfg = make_infer_raw_cfg(checkpoint=str(tmp_path / "best.pth"))
    monkeypatch.setattr(infer_script, "run_evaluation", lambda cfg: calls.setdefault("cfg", cfg))

    infer_script.main.__wrapped__(cfg)

    assert calls["cfg"] is cfg


def test_run_training_smoke(monkeypatch):
    import src.app.train as train_app

    calls = {}

    class FakeEngine:
        def __init__(self, **kwargs):
            calls["engine_init"] = kwargs

        def fit(self, **kwargs):
            calls["fit"] = kwargs
            return {"train_loss": [], "val_psnr": [], "val_ssim": [], "val_mae": []}

    monkeypatch.setattr(train_app, "Engine", FakeEngine)
    monkeypatch.setattr(train_app, "init_logger", lambda: None)
    monkeypatch.setattr(train_app, "get_logger", lambda: DummyLogger())
    monkeypatch.setattr(train_app, "is_main_process", lambda: True)
    monkeypatch.setattr(train_app, "get_rank", lambda: 0)
    monkeypatch.setattr(train_app, "get_world_size", lambda: 1)
    monkeypatch.setattr(train_app.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(train_app, "validate_train_runtime_inputs", lambda cfg: None)
    monkeypatch.setattr(train_app, "build_model_bundle", lambda cfg: {"model": DummyModel(), "model_name": "EDSR"})
    monkeypatch.setattr(train_app, "build_train_loaders", lambda cfg: ("train_loader", "val_loader"))
    monkeypatch.setattr(train_app, "build_optimizer", lambda cfg, model: object())
    monkeypatch.setattr(train_app, "build_scheduler", lambda cfg, optimizer: None)
    monkeypatch.setattr(train_app, "build_loss_fn", lambda cfg: object())
    monkeypatch.setattr(train_app, "build_train_callbacks", lambda *args, **kwargs: ["progress"])

    cfg = load_train_config(make_train_raw_cfg())
    train_app.run_training(cfg)

    assert calls["engine_init"]["ddp"] is False
    assert calls["fit"]["train_loader"] == "train_loader"
    assert calls["fit"]["val_loader"] == "val_loader"
    assert calls["fit"]["epochs"] == 2


def test_run_training_resume_uses_remaining_epochs(monkeypatch, tmp_path):
    import src.app.train as train_app

    calls = {}
    checkpoint_path = tmp_path / "last.pth"
    save_checkpoint_with_config(
        checkpoint_path,
        {
            "models": {"name": "EDSR", "params": {}},
            "dataset": {"param": "wind", "upscale": 2, "channels": 1},
        },
    )

    class FakeEngine:
        def __init__(self, **kwargs):
            calls["engine_init"] = kwargs

        def load_checkpoint(self, path, load_optimizer=True, load_scheduler=True, load_rng_state=True):
            calls["load_checkpoint"] = {
                "path": path,
                "load_optimizer": load_optimizer,
                "load_scheduler": load_scheduler,
                "load_rng_state": load_rng_state,
            }
            return 1

        def fit(self, **kwargs):
            calls["fit"] = kwargs
            return {"train_loss": [], "val_psnr": [], "val_ssim": [], "val_mae": []}

    monkeypatch.setattr(train_app, "Engine", FakeEngine)
    monkeypatch.setattr(train_app, "init_logger", lambda: None)
    monkeypatch.setattr(train_app, "get_logger", lambda: DummyLogger())
    monkeypatch.setattr(train_app, "is_main_process", lambda: True)
    monkeypatch.setattr(train_app, "get_rank", lambda: 0)
    monkeypatch.setattr(train_app, "get_world_size", lambda: 1)
    monkeypatch.setattr(train_app.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(train_app, "validate_train_runtime_inputs", lambda cfg: None)
    monkeypatch.setattr(train_app, "build_model_bundle", lambda cfg: {"model": DummyModel(), "model_name": "EDSR"})
    monkeypatch.setattr(train_app, "build_train_loaders", lambda cfg: ("train_loader", "val_loader"))
    monkeypatch.setattr(train_app, "build_optimizer", lambda cfg, model: object())
    monkeypatch.setattr(train_app, "build_scheduler", lambda cfg, optimizer: None)
    monkeypatch.setattr(train_app, "build_loss_fn", lambda cfg: object())
    monkeypatch.setattr(train_app, "build_train_callbacks", lambda *args, **kwargs: ["progress"])

    raw_cfg = make_train_raw_cfg()
    raw_cfg.resume = {
        "checkpoint": str(checkpoint_path),
        "load_optimizer": True,
        "load_scheduler": False,
        "load_rng_state": True,
    }
    cfg = load_train_config(raw_cfg)
    train_app.run_training(cfg)

    assert Path(calls["load_checkpoint"]["path"]) == checkpoint_path
    assert calls["load_checkpoint"]["load_optimizer"] is True
    assert calls["load_checkpoint"]["load_scheduler"] is False
    assert calls["load_checkpoint"]["load_rng_state"] is True
    assert calls["fit"]["epochs"] == 1
    assert calls["fit"]["start_epoch"] == 2


def test_run_training_resume_rejects_incompatible_checkpoint(monkeypatch, tmp_path):
    import src.app.train as train_app

    checkpoint_path = tmp_path / "last.pth"
    save_checkpoint_with_config(
        checkpoint_path,
        {
            "models": {"name": "RCAN", "params": {}},
            "dataset": {"param": "wind", "upscale": 2, "channels": 1},
        },
    )

    class FakeEngine:
        def __init__(self, **kwargs):
            pass

        def load_checkpoint(self, *args, **kwargs):
            raise AssertionError("不应在不兼容 checkpoint 上继续加载")

    monkeypatch.setattr(train_app, "Engine", FakeEngine)
    monkeypatch.setattr(train_app, "init_logger", lambda: None)
    monkeypatch.setattr(train_app, "get_logger", lambda: DummyLogger())
    monkeypatch.setattr(train_app, "is_main_process", lambda: True)
    monkeypatch.setattr(train_app, "get_rank", lambda: 0)
    monkeypatch.setattr(train_app, "get_world_size", lambda: 1)
    monkeypatch.setattr(train_app.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(train_app, "validate_train_runtime_inputs", lambda cfg: None)
    monkeypatch.setattr(train_app, "build_model_bundle", lambda cfg: {"model": DummyModel(), "model_name": "EDSR"})
    monkeypatch.setattr(train_app, "build_train_loaders", lambda cfg: ("train_loader", "val_loader"))
    monkeypatch.setattr(train_app, "build_optimizer", lambda cfg, model: object())
    monkeypatch.setattr(train_app, "build_scheduler", lambda cfg, optimizer: None)
    monkeypatch.setattr(train_app, "build_loss_fn", lambda cfg: object())
    monkeypatch.setattr(train_app, "build_train_callbacks", lambda *args, **kwargs: ["progress"])

    raw_cfg = make_train_raw_cfg()
    raw_cfg.resume = {
        "checkpoint": str(checkpoint_path),
        "load_optimizer": True,
        "load_scheduler": True,
        "load_rng_state": True,
    }
    cfg = load_train_config(raw_cfg)

    try:
        train_app.run_training(cfg)
    except ValueError as exc:
        assert "检查点配置与当前配置不一致" in str(exc)
    else:
        raise AssertionError("预期 run_training 在 checkpoint 不兼容时失败")


def test_run_evaluation_smoke(monkeypatch, tmp_path):
    import src.app.eval as eval_app

    calls = {}
    checkpoint_path = tmp_path / "best.pth"
    save_checkpoint_with_config(
        checkpoint_path,
        {
            "models": {"name": "EDSR", "params": {}},
            "dataset": {"param": "wind", "upscale": 2, "channels": 1},
        },
    )

    class FakeEvaluator:
        def __init__(self, model, callbacks=None):
            calls["evaluator_init"] = {"model": model, "callbacks": callbacks}

        def load_checkpoint(self, path):
            calls["checkpoint"] = path

        def run(self, **kwargs):
            calls["run"] = kwargs
            return {"psnr": 30.0, "ssim": 0.9, "mae": 0.1, "max_mae": 0.2}

    monkeypatch.setattr(eval_app, "Evaluator", FakeEvaluator)
    monkeypatch.setattr(eval_app, "init_logger", lambda: None)
    monkeypatch.setattr(eval_app, "get_logger", lambda: DummyLogger())
    monkeypatch.setattr(eval_app, "validate_eval_runtime_inputs", lambda cfg: None)
    monkeypatch.setattr(eval_app, "build_model_bundle", lambda cfg: {"model": DummyModel(), "model_name": "EDSR"})
    monkeypatch.setattr(eval_app, "build_eval_callbacks", lambda cfg: ["metrics", "progress"])
    monkeypatch.setattr(eval_app, "build_test_loader", lambda cfg: ["sample"])
    monkeypatch.setattr(eval_app, "load_eval_mask", lambda *args, **kwargs: None)

    cfg = load_evaluate_config(make_evaluate_raw_cfg(checkpoint=str(checkpoint_path)))
    eval_app.run_evaluation(cfg)

    assert Path(calls["checkpoint"]) == checkpoint_path
    assert calls["run"]["test_loader"] == ["sample"]
    assert calls["run"]["mean"] == [0.0]
    assert calls["run"]["std"] == [1.0]


def test_run_inference_smoke(monkeypatch, tmp_path):
    import src.app.eval as eval_app

    calls = {}
    checkpoint_path = tmp_path / "best.pth"
    save_checkpoint_with_config(
        checkpoint_path,
        {
            "models": {"name": "EDSR", "params": {}},
            "dataset": {"param": "wind", "upscale": 2, "channels": 1},
        },
    )

    class FakeEvaluator:
        def __init__(self, model, callbacks=None):
            calls["evaluator_init"] = {"model": model, "callbacks": callbacks}

        def load_checkpoint(self, path):
            calls["checkpoint"] = path

        def run(self, **kwargs):
            calls["run"] = kwargs
            return {}

    monkeypatch.setattr(eval_app, "Evaluator", FakeEvaluator)
    monkeypatch.setattr(eval_app, "init_logger", lambda: None)
    monkeypatch.setattr(eval_app, "get_logger", lambda: DummyLogger())
    monkeypatch.setattr(eval_app, "validate_eval_runtime_inputs", lambda cfg: None)
    monkeypatch.setattr(eval_app, "build_model_bundle", lambda cfg: {"model": DummyModel(), "model_name": "EDSR"})
    monkeypatch.setattr(eval_app, "build_eval_callbacks", lambda cfg: ["save_results"])
    monkeypatch.setattr(eval_app, "build_test_loader", lambda cfg: ["sample"])
    monkeypatch.setattr(eval_app, "load_eval_mask", lambda *args, **kwargs: "mask")

    cfg = load_infer_config(make_infer_raw_cfg(checkpoint=str(checkpoint_path)))
    eval_app.run_evaluation(cfg)

    assert Path(calls["checkpoint"]) == checkpoint_path
    assert calls["run"]["test_loader"] == ["sample"]
    assert calls["run"]["mask"] is None


def test_run_evaluation_rejects_incompatible_checkpoint(monkeypatch, tmp_path):
    import src.app.eval as eval_app

    checkpoint_path = tmp_path / "best.pth"
    save_checkpoint_with_config(
        checkpoint_path,
        {
            "models": {"name": "EDSR", "params": {}},
            "dataset": {"param": "salinity", "upscale": 2, "channels": 1},
        },
    )

    class FakeEvaluator:
        def __init__(self, model, callbacks=None):
            pass

        def load_checkpoint(self, path):
            raise AssertionError("不应在不兼容 checkpoint 上继续加载")

        def run(self, **kwargs):
            raise AssertionError("不应运行 evaluator")

    monkeypatch.setattr(eval_app, "Evaluator", FakeEvaluator)
    monkeypatch.setattr(eval_app, "init_logger", lambda: None)
    monkeypatch.setattr(eval_app, "get_logger", lambda: DummyLogger())
    monkeypatch.setattr(eval_app, "validate_eval_runtime_inputs", lambda cfg: None)
    monkeypatch.setattr(eval_app, "build_model_bundle", lambda cfg: {"model": DummyModel(), "model_name": "EDSR"})
    monkeypatch.setattr(eval_app, "build_eval_callbacks", lambda cfg: ["metrics", "progress"])
    monkeypatch.setattr(eval_app, "build_test_loader", lambda cfg: ["sample"])
    monkeypatch.setattr(eval_app, "load_eval_mask", lambda *args, **kwargs: None)

    cfg = load_evaluate_config(make_evaluate_raw_cfg(checkpoint=str(checkpoint_path)))

    try:
        eval_app.run_evaluation(cfg)
    except ValueError as exc:
        assert "检查点配置与当前配置不一致" in str(exc)
    else:
        raise AssertionError("预期 run_evaluation 在 checkpoint 不兼容时失败")
