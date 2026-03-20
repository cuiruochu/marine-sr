"""Dataset and dataloader builders."""

from __future__ import annotations

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.datasets import MarineEvalDataset, MarineInferenceDataset, MarineTrainDataset
from src.utils.distributed import is_distributed


def build_train_dataset(spec) -> MarineTrainDataset:
    return MarineTrainDataset(
        lr_root=spec.train_lr_root,
        hr_root=spec.train_hr_root,
        upscale=spec.upscale,
        lr_patch_size=spec.lr_patch_size,
        mean=spec.mean,
        std=spec.std,
    )


def build_val_dataset(spec, max_sample: int | bool | None = None) -> MarineEvalDataset:
    val_max_sample = spec.max_sample if max_sample is None else max_sample
    return MarineEvalDataset(
        lr_root=spec.val_lr_root,
        hr_root=spec.val_hr_root,
        upscale=spec.upscale,
        mean=spec.mean,
        std=spec.std,
        max_sample=val_max_sample,
    )


def build_test_dataset(spec):
    if getattr(spec, "mode", None) == "inference":
        return MarineInferenceDataset(
            lr_root=spec.lr_root,
            mean=spec.mean,
            std=spec.std,
        )

    return MarineEvalDataset(
        lr_root=spec.lr_root,
        hr_root=spec.hr_root,
        upscale=spec.upscale,
        mean=spec.mean,
        std=spec.std,
        max_sample=getattr(spec, "max_sample", False),
    )


def build_train_dataloader(dataset, batch_size: int, num_workers: int) -> DataLoader:
    sampler = DistributedSampler(dataset, shuffle=True) if is_distributed() else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        drop_last=False,
    )


def build_eval_dataloader(dataset, batch_size: int, num_workers: int) -> DataLoader:
    sampler = DistributedSampler(dataset, shuffle=False) if is_distributed() else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        drop_last=False,
    )


def build_train_val_loaders(
    spec,
    *,
    train_batch_size: int,
    train_num_workers: int,
    eval_num_workers: int,
):
    train_dataset = build_train_dataset(spec)
    val_dataset = build_val_dataset(spec)
    train_loader = build_train_dataloader(
        train_dataset,
        batch_size=train_batch_size,
        num_workers=train_num_workers,
    )
    val_loader = build_eval_dataloader(
        val_dataset,
        batch_size=1,
        num_workers=eval_num_workers,
    )
    return train_loader, val_loader


def build_test_loader(spec, batch_size: int, num_workers: int) -> DataLoader:
    dataset = build_test_dataset(spec)
    return build_eval_dataloader(dataset, batch_size=batch_size, num_workers=num_workers)


def set_epoch_for_sampler(loader, epoch: int) -> None:
    sampler = getattr(loader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
