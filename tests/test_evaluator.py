import uuid
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir

from src.app.config_parsing import load_evaluate_config
from src.app.eval import run_eval_task
from src.callbacks.save_results import SaveResultsCallback
from src.core.callbacks import Callback
from src.core.evaluator import Evaluator

CONFIG_DIR = str((Path(__file__).resolve().parents[1] / "configs").resolve())


class IdentityModel(nn.Module):
    def forward(self, x):
        return x


def _make_test_root() -> Path:
    root = Path.cwd() / ".test-artifacts" / f"evaluator-{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _compose(config_name: str):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name=config_name)


def test_evaluator_supports_masked_multi_channel_metrics_and_masked_save():
    tmp_path = _make_test_root()
    model = IdentityModel()
    callback = SaveResultsCallback(save_dir=str(tmp_path))
    evaluator = Evaluator(model=model, callbacks=[callback], device="cpu")

    hr = torch.rand(1, 2, 4, 4)
    loader = [(hr.clone(), hr.clone(), "sample.npy")]
    mask = np.array(
        [
            [True, False, True, False],
            [True, True, False, False],
            [False, False, True, True],
            [True, False, True, False],
        ],
        dtype=bool,
    )

    logs = evaluator.run(
        test_loader=loader,
        mean=[0.0, 0.0],
        std=[1.0, 1.0],
        mask=mask,
    )

    assert logs["psnr"] == float("inf")
    assert pytest.approx(logs["ssim"], rel=1e-6, abs=1e-6) == 1.0
    assert logs["mae"] == 0.0
    assert logs["max_mae"] == 0.0

    saved = np.load(tmp_path / "sample.npy", allow_pickle=True)
    assert saved.shape == (2, 4, 4)
    assert saved.dtype == object
    assert saved[0, 0, 1] is None
    assert saved[1, 3, 1] is None
    assert saved[0, 0, 0] is not None


def test_evaluator_supports_batch_size_gt_one_and_saves_each_sample():
    model = IdentityModel()
    tmp_path = _make_test_root()
    callback = SaveResultsCallback(save_dir=str(tmp_path))
    evaluator = Evaluator(model=model, device="cpu", callbacks=[callback])

    hr = torch.rand(2, 1, 4, 4)
    loader = [(hr.clone(), hr.clone(), ["sample_a.npy", "sample_b.npy"])]

    logs = evaluator.run(
        test_loader=loader,
        mean=[0.0],
        std=[1.0],
    )

    assert logs["psnr"] == float("inf")
    assert pytest.approx(logs["ssim"], rel=1e-6, abs=1e-6) == 1.0
    assert logs["mae"] == 0.0
    assert logs["max_mae"] == 0.0
    assert (tmp_path / "sample_a.npy").exists()
    assert (tmp_path / "sample_b.npy").exists()


def test_evaluator_aggregates_mean_metrics_by_sample_count(monkeypatch):
    model = IdentityModel()
    evaluator = Evaluator(model=model, device="cpu")

    loader = [
        (torch.ones(2, 1, 4, 4), torch.ones(2, 1, 4, 4), ["sample_a.npy", "sample_b.npy"]),
        (torch.ones(1, 1, 4, 4), torch.ones(1, 1, 4, 4), ["sample_c.npy"]),
    ]

    def fake_psnr(img1, img2, mask=None):
        return torch.tensor([10.0, 20.0]) if img1.shape[0] == 2 else torch.tensor([0.0])

    def fake_ssim(img1, img2, mask=None):
        return torch.tensor([0.6, 0.3]) if img1.shape[0] == 2 else torch.tensor([0.0])

    def fake_mae(img1, img2, mask=None):
        return torch.tensor([4.0, 5.0]) if img1.shape[0] == 2 else torch.tensor([1.0])

    def fake_max_mae(img1, img2, mask=None):
        return torch.tensor([7.0, 2.0]) if img1.shape[0] == 2 else torch.tensor([3.0])

    monkeypatch.setattr("src.core.evaluator.calculate_psnr", fake_psnr)
    monkeypatch.setattr("src.core.evaluator.calculate_ssim", fake_ssim)
    monkeypatch.setattr("src.core.evaluator.calculate_mae", fake_mae)
    monkeypatch.setattr("src.core.evaluator.calculate_max_mae", fake_max_mae)

    logs = evaluator.run(
        test_loader=loader,
        mean=[0.0],
        std=[1.0],
    )

    assert pytest.approx(logs["psnr"], rel=1e-6, abs=1e-6) == 10.0
    assert pytest.approx(logs["ssim"], rel=1e-6, abs=1e-6) == 0.3
    assert pytest.approx(logs["mae"], rel=1e-6, abs=1e-6) == 10.0 / 3.0
    assert logs["max_mae"] == 7.0


def test_evaluator_passes_per_sample_metric_tensors_to_callbacks():
    class CaptureMetricsCallback(Callback):
        def __init__(self):
            self.metrics = []

        def on_batch_end(self, evaluator, batch, logs=None, **kwargs):
            self.metrics.append(kwargs["metrics"])

    callback = CaptureMetricsCallback()
    evaluator = Evaluator(model=IdentityModel(), device="cpu", callbacks=[callback])

    hr = torch.tensor(
        [
            [[[0.0, 0.0], [0.0, 0.0]]],
            [[[1.0, 1.0], [1.0, 1.0]]],
        ]
    )
    sr = hr.clone()
    sr[1] = 0.0
    loader = [(sr, hr, ["sample_a.npy", "sample_b.npy"])]

    evaluator.run(
        test_loader=loader,
        mean=[0.0],
        std=[1.0],
    )

    assert len(callback.metrics) == 1
    batch_metrics = callback.metrics[0]
    assert set(batch_metrics) == {"psnr", "ssim", "mae", "max_mae"}
    for values in batch_metrics.values():
        assert isinstance(values, torch.Tensor)
        assert values.shape == (2,)


def test_run_eval_task_skips_checkpoint_loading_for_bicubic(monkeypatch):
    raw_cfg = _compose("wind/evaluate_x2")
    raw_cfg.models.name = "Bicubic"
    raw_cfg.evaluate.checkpoint = None
    cfg = load_evaluate_config(raw_cfg)

    class StubEvaluator:
        instance = None

        def __init__(self, model, callbacks, device=None):
            del device
            self.model = model
            self.callbacks = callbacks
            self.loaded_checkpoint = None
            self.run_kwargs = None
            StubEvaluator.instance = self

        def load_checkpoint(self, path):
            self.loaded_checkpoint = path

        def run(self, **kwargs):
            self.run_kwargs = kwargs
            return {}

    monkeypatch.setattr("src.app.eval.init_task_logger", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("src.app.eval.validate_eval_runtime_inputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "src.app.eval.validate_checkpoint_matches_config",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not validate checkpoint")),
    )
    monkeypatch.setattr("src.app.eval.build_model_bundle", lambda *_args, **_kwargs: {"model": IdentityModel()})
    monkeypatch.setattr("src.app.eval.build_eval_callbacks", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("src.app.eval.build_test_loader", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("src.app.eval._load_mask", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("src.app.eval.Evaluator", StubEvaluator)

    run_eval_task(cfg)

    assert StubEvaluator.instance is not None
    assert StubEvaluator.instance.loaded_checkpoint is None
    assert StubEvaluator.instance.run_kwargs["mode"] == "evaluation"
