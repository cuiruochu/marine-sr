"""训练数据集与加载器装配器。"""

from src.datasets.marine import MarineEvalDataset, MarineTrainDataset
from src.datasets.samplers import ChannelBatchSampler


def build_train_dataset(spec):
    """构建训练数据集。"""
    pair = spec.train_pair
    return MarineTrainDataset(
        lr_root=pair.lr_root,
        hr_root=pair.hr_root,
        upscale=spec.upscale,
        lr_patch_size=spec.lr_patch_size,
        mean=pair.normalize.mean,
        std=pair.normalize.std,
    )


def build_val_dataset(spec):
    """构建训练验证数据集。验证损失应与训练损失处于同一归一化空间。"""
    return MarineEvalDataset(
        lr_root=spec.val_pair.lr_root,
        hr_root=spec.val_pair.hr_root,
        upscale=spec.upscale,
        mean=spec.val_pair.normalize.mean,
        std=spec.val_pair.normalize.std,
        sample_limit=spec.val_pair.max_sample if spec.val_pair.max_sample else False,
        normalize_hr=True,
    )


def build_train_val_loaders(
    spec,
    train_batch_size: int,
    train_num_workers: int,
    eval_num_workers: int,
):
    """构建训练和验证数据加载器。"""
    from .loaders import build_eval_dataloader, build_train_dataloader

    train_set = build_train_dataset(spec)
    batch_sampler = ChannelBatchSampler(train_set, batch_size=train_batch_size, drop_last=True)
    train_loader = build_train_dataloader(
        train_set,
        num_workers=train_num_workers,
        batch_sampler=batch_sampler,
    )

    val_set = build_val_dataset(spec)
    val_loader = build_eval_dataloader(
        val_set,
        batch_size=1,
        num_workers=eval_num_workers,
    )

    return train_loader, val_loader
