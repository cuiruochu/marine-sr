"""Dataloader worker persistence tests."""

import torch

from src.data.loaders import build_eval_dataloader, build_train_dataloader


class DummyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        x = torch.zeros(1, 8, 8)
        y = torch.ones(1, 16, 16)
        return x, y


def test_train_dataloader_disables_persistent_workers_when_num_workers_zero():
    loader = build_train_dataloader(DummyDataset(), batch_size=2, num_workers=0)

    assert loader.num_workers == 0
    assert loader.persistent_workers is False


def test_train_dataloader_enables_persistent_workers_when_num_workers_positive():
    loader = build_train_dataloader(DummyDataset(), batch_size=2, num_workers=1)

    assert loader.num_workers == 1
    assert loader.persistent_workers is True


def test_eval_dataloader_matches_persistent_worker_policy():
    loader = build_eval_dataloader(DummyDataset(), batch_size=1, num_workers=1)

    assert loader.num_workers == 1
    assert loader.persistent_workers is True
