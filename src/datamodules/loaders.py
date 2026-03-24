"""
数据加载器构建
"""

import torch
from torch.utils.data import DataLoader


def build_train_dataloader(
    dataset,
    num_workers: int,
    batch_size: int | None = None,
    batch_sampler=None,
    shuffle: bool = True,
) -> DataLoader:
    """构建训练数据加载器。"""
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0

    if batch_sampler is not None:
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )


def build_eval_dataloader(
    dataset,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    """构建评估数据加载器。"""
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
