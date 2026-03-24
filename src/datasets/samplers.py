"""数据采样器。"""

from __future__ import annotations

import random

from torch.utils.data.sampler import BatchSampler


class ChannelBatchSampler(BatchSampler):
    """按 channel_count 分桶并组成批次。"""

    def __init__(self, dataset, batch_size: int, drop_last: bool = True, shuffle: bool = True):
        if batch_size <= 0:
            raise ValueError(f"batch_size 必须为正整数: {batch_size}")
        if not hasattr(dataset, "channel_counts"):
            raise ValueError("dataset 必须暴露 channel_counts 供 ChannelBatchSampler 分桶")

        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        self.channel_buckets = {}
        for index, channel_count in enumerate(dataset.channel_counts):
            self.channel_buckets.setdefault(int(channel_count), []).append(index)

    def __iter__(self):
        bucket_indices = {key: list(indices) for key, indices in self.channel_buckets.items()}

        if self.shuffle:
            for indices in bucket_indices.values():
                random.shuffle(indices)

        all_batches = []
        for indices in bucket_indices.values():
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) == self.batch_size:
                    all_batches.append(batch)
                elif not self.drop_last and batch:
                    all_batches.append(batch)

        if self.shuffle:
            random.shuffle(all_batches)

        return iter(all_batches)

    def __len__(self):
        total = 0
        for indices in self.channel_buckets.values():
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += (len(indices) + self.batch_size - 1) // self.batch_size
        return total
