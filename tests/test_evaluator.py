import uuid
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.callbacks.save_results import SaveResultsCallback
from src.core.evaluator import Evaluator


class IdentityModel(nn.Module):
    def forward(self, x):
        return x


def _make_test_root() -> Path:
    root = Path.cwd() / ".test-artifacts" / f"evaluator-{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    return root


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
        return 10.0 if img1.shape[0] == 2 else 0.0

    def fake_ssim(img1, img2, mask=None):
        return 0.6 if img1.shape[0] == 2 else 0.0

    def fake_mae(img1, img2, mask=None):
        return 4.0 if img1.shape[0] == 2 else 1.0

    def fake_max_mae(img1, img2, mask=None):
        return 7.0 if img1.shape[0] == 2 else 3.0

    monkeypatch.setattr("src.core.evaluator.calculate_psnr", fake_psnr)
    monkeypatch.setattr("src.core.evaluator.calculate_ssim", fake_ssim)
    monkeypatch.setattr("src.core.evaluator.calculate_mae", fake_mae)
    monkeypatch.setattr("src.core.evaluator.calculate_max_mae", fake_max_mae)

    logs = evaluator.run(
        test_loader=loader,
        mean=[0.0],
        std=[1.0],
    )

    assert pytest.approx(logs["psnr"], rel=1e-6, abs=1e-6) == 20.0 / 3.0
    assert pytest.approx(logs["ssim"], rel=1e-6, abs=1e-6) == 0.4
    assert pytest.approx(logs["mae"], rel=1e-6, abs=1e-6) == 3.0
    assert logs["max_mae"] == 7.0
