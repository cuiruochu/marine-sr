"""测试数据集。"""

import os
import tempfile

import numpy as np
from torch.utils.data import ConcatDataset

from src.datasets.marine import MarineEvalDataset, MarineInferDataset, MarineTrainDataset
from src.datasets.samplers import ChannelBatchSampler


def _write_pair_samples(lr_dir, hr_dir, count, *, lr_shape, hr_shape):
    os.makedirs(lr_dir, exist_ok=True)
    os.makedirs(hr_dir, exist_ok=True)
    for i in range(count):
        np.save(os.path.join(lr_dir, f"sample_{i}.npy"), np.random.rand(*lr_shape).astype(np.float32))
        np.save(os.path.join(hr_dir, f"sample_{i}.npy"), np.random.rand(*hr_shape).astype(np.float32))


class TestMarineTrainDataset:
    def test_paired_sample_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lr_dir = os.path.join(tmpdir, "lr")
            hr_dir = os.path.join(tmpdir, "hr")
            _write_pair_samples(lr_dir, hr_dir, 3, lr_shape=(64, 64), hr_shape=(128, 128))

            dataset = MarineTrainDataset(
                lr_root=lr_dir,
                hr_root=hr_dir,
                upscale=2,
                lr_patch_size=16,
                mean=[0.5],
                std=[1.0],
            )

            assert len(dataset) == 3
            assert dataset.channel_count == 1

            lr, hr = dataset[0]
            assert lr.shape == (1, 16, 16)
            assert hr.shape == (1, 32, 32)


class TestMarineEvalDataset:
    def test_paired_eval_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lr_dir = os.path.join(tmpdir, "lr")
            hr_dir = os.path.join(tmpdir, "hr")
            _write_pair_samples(lr_dir, hr_dir, 3, lr_shape=(64, 64), hr_shape=(128, 128))

            dataset = MarineEvalDataset(
                lr_root=lr_dir,
                hr_root=hr_dir,
                upscale=2,
                mean=[0.5],
                std=[1.0],
                sample_limit=3,
                return_filename=True,
            )

            lr, hr, filename = dataset[0]
            assert lr.shape == (1, 64, 64)
            assert hr.shape == (1, 128, 128)
            assert filename.endswith(".npy")

    def test_eval_dataset_can_normalize_hr_for_validation_loss(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lr_dir = os.path.join(tmpdir, "lr")
            hr_dir = os.path.join(tmpdir, "hr")
            os.makedirs(lr_dir, exist_ok=True)
            os.makedirs(hr_dir, exist_ok=True)
            np.save(os.path.join(lr_dir, "sample_0.npy"), np.full((4, 4), 3.0, dtype=np.float32))
            np.save(os.path.join(hr_dir, "sample_0.npy"), np.full((8, 8), 7.0, dtype=np.float32))

            dataset = MarineEvalDataset(
                lr_root=lr_dir,
                hr_root=hr_dir,
                upscale=2,
                mean=[1.0],
                std=[2.0],
                sample_limit=1,
                return_filename=False,
                normalize_hr=True,
            )

            lr, hr = dataset[0]
            assert float(lr[0, 0, 0]) == 1.0
            assert float(hr[0, 0, 0]) == 3.0


class TestMarineInferDataset:
    def test_infer_dataset_returns_normalized_tensor_and_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lr_dir = os.path.join(tmpdir, "lr")
            os.makedirs(lr_dir, exist_ok=True)
            np.save(os.path.join(lr_dir, "sample_0.npy"), np.full((4, 4), 3.0, dtype=np.float32))

            dataset = MarineInferDataset(
                lr_root=lr_dir,
                upscale=2,
                mean=[1.0],
                std=[2.0],
                sample_limit=1,
                return_filename=True,
            )

            lr, filename = dataset[0]
            assert lr.shape == (1, 4, 4)
            assert float(lr[0, 0, 0]) == 1.0
            assert filename == "sample_0.npy"


class TestChannelBatchSampler:
    def test_batch_sampler(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lr_dir_1 = os.path.join(tmpdir, "lr1")
            hr_dir_1 = os.path.join(tmpdir, "hr1")
            lr_dir_2 = os.path.join(tmpdir, "lr2")
            hr_dir_2 = os.path.join(tmpdir, "hr2")
            _write_pair_samples(lr_dir_1, hr_dir_1, 10, lr_shape=(64, 64), hr_shape=(128, 128))
            _write_pair_samples(lr_dir_2, hr_dir_2, 9, lr_shape=(2, 64, 64), hr_shape=(2, 128, 128))

            dataset_1 = MarineTrainDataset(
                lr_root=lr_dir_1,
                hr_root=hr_dir_1,
                upscale=2,
                lr_patch_size=16,
                mean=[0.5],
                std=[1.0],
            )
            dataset_2 = MarineTrainDataset(
                lr_root=lr_dir_2,
                hr_root=hr_dir_2,
                upscale=2,
                lr_patch_size=16,
                mean=[0.5, 0.5],
                std=[1.0, 1.0],
            )
            dataset = ConcatDataset([dataset_1, dataset_2])
            dataset.channel_counts = dataset_1.channel_counts + dataset_2.channel_counts

            sampler = ChannelBatchSampler(dataset, batch_size=4, drop_last=True, shuffle=False)
            batches = list(sampler)

            assert len(batches) > 0
            for batch in batches:
                assert len(batch) == 4
                channels = {dataset.channel_counts[index] for index in batch}
                assert len(channels) == 1
