"""
数据集与数据加载测试
"""

import os
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.data import (
    build_eval_dataloader,
    build_test_dataset,
    build_test_loader,
    build_train_dataloader,
    build_train_dataset,
    build_val_dataset,
)
from src.datasets import MarineEvalDataset, MarineInferenceDataset, MarineTrainDataset
from src.datasets.utils import normalize


class TestDatasetTransforms:
    def test_normalize_shape(self):
        mean = torch.tensor([1.0]).unsqueeze(-1).unsqueeze(-1)
        std = torch.tensor([2.0]).unsqueeze(-1).unsqueeze(-1)
        x = torch.randn(1, 64, 64)
        normalized = normalize(mean, std, x)
        assert normalized.shape == x.shape


class TestMarineEvalDataset:
    @pytest.fixture
    def temp_data_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lr_dir = os.path.join(tmpdir, "lr")
            hr_dir = os.path.join(tmpdir, "hr")
            os.makedirs(lr_dir)
            os.makedirs(hr_dir)
            for i in range(3):
                lr = np.random.rand(32, 32).astype(np.float32)
                hr = np.random.rand(64, 64).astype(np.float32)
                np.save(os.path.join(lr_dir, f"test_{i}.npy"), lr)
                np.save(os.path.join(hr_dir, f"test_{i}.npy"), hr)
            yield lr_dir, hr_dir

    def test_eval_dataset_length(self, temp_data_dir):
        lr_dir, hr_dir = temp_data_dir
        dataset = MarineEvalDataset(
            lr_root=lr_dir,
            hr_root=hr_dir,
            upscale=2,
            mean=[0.5],
            std=[1.0],
            max_sample=False,
        )
        assert len(dataset) == 3

    def test_eval_dataset_getitem_shape(self, temp_data_dir):
        lr_dir, hr_dir = temp_data_dir
        dataset = MarineEvalDataset(
            lr_root=lr_dir,
            hr_root=hr_dir,
            upscale=2,
            mean=[0.5],
            std=[1.0],
            max_sample=False,
        )
        lr, hr, filename = dataset[0]
        assert hr.shape == (1, 64, 64)
        assert lr.shape == (1, 32, 32)
        assert filename.endswith(".npy")


class TestMarineInferenceDataset:
    def test_inference_dataset_getitem_shape(self, tmp_path):
        for i in range(3):
            lr = np.random.rand(32, 32).astype(np.float32)
            np.save(tmp_path / f"sample_{i}.npy", lr)

        dataset = MarineInferenceDataset(
            lr_root=str(tmp_path),
            mean=[0.5],
            std=[1.0],
        )

        lr, filename = dataset[0]
        assert lr.shape == (1, 32, 32)
        assert filename.endswith(".npy")


class TestMarineTrainDataset:
    @pytest.fixture
    def temp_data_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lr_dir = os.path.join(tmpdir, "lr")
            hr_dir = os.path.join(tmpdir, "hr")
            os.makedirs(lr_dir)
            os.makedirs(hr_dir)
            for i in range(5):
                lr = np.random.rand(60, 60).astype(np.float32)
                hr = np.random.rand(120, 120).astype(np.float32)
                np.save(os.path.join(lr_dir, f"train_{i}.npy"), lr)
                np.save(os.path.join(hr_dir, f"train_{i}.npy"), hr)
            yield lr_dir, hr_dir

    def test_train_dataset_length(self, temp_data_dir):
        lr_dir, hr_dir = temp_data_dir
        dataset = MarineTrainDataset(
            lr_root=lr_dir,
            hr_root=hr_dir,
            upscale=2,
            lr_patch_size=30,
            mean=[0.5],
            std=[1.0],
        )
        assert len(dataset) == 5

    def test_train_dataset_getitem_shape(self, temp_data_dir):
        lr_dir, hr_dir = temp_data_dir
        dataset = MarineTrainDataset(
            lr_root=lr_dir,
            hr_root=hr_dir,
            upscale=2,
            lr_patch_size=30,
            mean=[0.5],
            std=[1.0],
        )
        lr, hr = dataset[0]
        assert hr.shape == (1, 60, 60)
        assert lr.shape == (1, 30, 30)


class TestDataBuilders:
    @pytest.fixture
    def train_spec(self, tmp_path):
        train_lr_root = tmp_path / "Train" / "LR"
        train_hr_root = tmp_path / "Train" / "HR"
        val_lr_root = tmp_path / "Val" / "LR"
        val_hr_root = tmp_path / "Val" / "HR"
        eval_lr_root = tmp_path / "Eval" / "LR"
        eval_hr_root = tmp_path / "Eval" / "HR"
        infer_lr_root = tmp_path / "Infer" / "LR"

        for root in (train_lr_root, train_hr_root, val_lr_root, val_hr_root, eval_lr_root, eval_hr_root, infer_lr_root):
            root.mkdir(parents=True)

        for root, size in (
            (train_lr_root, 60),
            (val_lr_root, 60),
            (eval_lr_root, 60),
            (infer_lr_root, 60),
            (train_hr_root, 120),
            (val_hr_root, 120),
            (eval_hr_root, 120),
        ):
            for i in range(3):
                data = np.random.rand(size, size).astype(np.float32)
                np.save(root / f"sample_{i}.npy", data)

        train_spec = SimpleNamespace(
            upscale=2,
            lr_patch_size=30,
            batch_size=2,
            mean=[0.5],
            std=[1.0],
            train_lr_root=str(train_lr_root),
            train_hr_root=str(train_hr_root),
            val_lr_root=str(val_lr_root),
            val_hr_root=str(val_hr_root),
            max_sample=False,
        )

        eval_spec = SimpleNamespace(
            upscale=2,
            mean=[0.5],
            std=[1.0],
            lr_root=str(eval_lr_root),
            hr_root=str(eval_hr_root),
            mode="evaluation",
            max_sample=False,
        )

        infer_spec = SimpleNamespace(
            upscale=2,
            mean=[0.5],
            std=[1.0],
            lr_root=str(infer_lr_root),
            hr_root=None,
            mode="inference",
            max_sample=False,
        )

        return train_spec, eval_spec, infer_spec

    def test_build_train_dataset(self, train_spec):
        TrainSpec, _, _ = train_spec
        dataset = build_train_dataset(TrainSpec)
        assert isinstance(dataset, MarineTrainDataset)
        assert len(dataset) == 3

    def test_build_val_dataset(self, train_spec):
        TrainSpec, _, _ = train_spec
        dataset = build_val_dataset(TrainSpec, max_sample=2)
        assert isinstance(dataset, MarineEvalDataset)
        assert len(dataset) == 2

    def test_build_test_dataset(self, train_spec):
        _, EvalSpec, _ = train_spec
        dataset = build_test_dataset(EvalSpec)
        assert isinstance(dataset, MarineEvalDataset)
        assert len(dataset) == 3

    def test_build_inference_dataset(self, train_spec):
        _, _, InferSpec = train_spec
        dataset = build_test_dataset(InferSpec)
        assert isinstance(dataset, MarineInferenceDataset)
        assert len(dataset) == 3

    def test_build_train_dataloader(self, train_spec):
        TrainSpec, _, _ = train_spec
        dataset = build_train_dataset(TrainSpec)
        loader = build_train_dataloader(dataset, batch_size=2, num_workers=0)
        batch = next(iter(loader))
        assert batch[0].shape == (2, 1, 30, 30)
        assert batch[1].shape == (2, 1, 60, 60)
        assert loader.persistent_workers is False

    def test_build_eval_dataloader(self, train_spec):
        TrainSpec, _, _ = train_spec
        dataset = build_val_dataset(TrainSpec, max_sample=2)
        loader = build_eval_dataloader(dataset, batch_size=1, num_workers=0)
        batch = next(iter(loader))
        assert batch[0].shape == (1, 1, 60, 60)
        assert batch[1].shape == (1, 1, 120, 120)
        assert loader.persistent_workers is False

    def test_dataloader_enables_persistent_workers_when_workers_positive(self, train_spec):
        TrainSpec, _, _ = train_spec
        dataset = build_train_dataset(TrainSpec)
        loader = build_train_dataloader(dataset, batch_size=2, num_workers=1)

        assert loader.num_workers == 1
        assert loader.persistent_workers is True

    def test_build_test_loader(self, train_spec):
        _, EvalSpec, _ = train_spec
        loader = build_test_loader(EvalSpec, batch_size=1, num_workers=0)
        batch = next(iter(loader))
        assert batch[0].shape == (1, 1, 60, 60)
        assert batch[1].shape == (1, 1, 120, 120)
        assert loader.persistent_workers is False

    def test_build_inference_loader(self, train_spec):
        _, _, InferSpec = train_spec
        loader = build_test_loader(InferSpec, batch_size=1, num_workers=0)
        batch = next(iter(loader))
        assert batch[0].shape == (1, 1, 60, 60)
        assert batch[1][0].endswith(".npy")
