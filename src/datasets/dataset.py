import os
import torch
import random
import numpy as np
import torch.utils.data as data
from torchvision import transforms
from torch.utils.data.sampler import BatchSampler
from .utils import resize, patchify, augement, normalize, encode_mwd


def _is_mwd_param(marine_param: str) -> bool:
    """检查是否为 MWD 参数（2 通道编码）"""
    return marine_param.lower() == "mwd"


def get_loader(train_cfg, val_cfg, batch_size):
    """
    创建训练和验证数据加载器。

    Args:
        train_cfg: 训练配置（TrainConfig）
        val_cfg: 验证配置（TrainConfig）
        batch_size: 批次大小

    Note:
        参数验证已在 TrainConfig._validate() 中完成
    """
    marine_param = train_cfg.marine_param

    # unified training
    if train_cfg.unified:
        train_set = UniformTrainSet(
            hr_paths=train_cfg.train_hr_root,
            upscale=train_cfg.upscale,
            lr_patch_size=train_cfg.lr_patch_size,
            mean=train_cfg.mean,
            std=train_cfg.std
        )
        batch_sampler = UniformClassBatchSampler(train_set, batch_size=batch_size)
        train_loader = data.DataLoader(
            train_set,
            batch_sampler=batch_sampler,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True
        )

        val_set = MarineTestSet(
            hr_root=val_cfg.val_hr_root,
            upscale=val_cfg.upscale,
            mean=val_cfg.mean,
            std=val_cfg.std,
            is_mwd=_is_mwd_param(val_cfg.marine_param),
            sample_q=10
        )
        val_loader = data.DataLoader(val_set, 1, shuffle=False, num_workers=4,
                                     drop_last=False, persistent_workers=True, pin_memory=True)
        return train_loader, val_loader

    # single training - 配置验证已在 TrainConfig 中完成
    train_set = MarineTrainSet(
        hr_root=train_cfg.train_hr_root,
        upscale=train_cfg.upscale,
        lr_patch_size=train_cfg.lr_patch_size,
        mean=train_cfg.mean,
        std=train_cfg.std,
        is_mwd=_is_mwd_param(train_cfg.marine_param),
    )
    val_set = MarineTestSet(
        hr_root=val_cfg.val_hr_root,
        upscale=val_cfg.upscale,
        mean=val_cfg.mean,
        std=val_cfg.std,
        is_mwd=_is_mwd_param(val_cfg.marine_param),
        sample_q=10
    )
    train_loader = data.DataLoader(
        train_set,
        batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        persistent_workers=True,
        pin_memory=True
    )
    val_loader = data.DataLoader(val_set, 1, shuffle=False, num_workers=4,
                                 drop_last=False, persistent_workers=True, pin_memory=True)
    return train_loader, val_loader


def get_test_loader(config):
    """
    创建测试数据加载器。

    Args:
        config: 测试配置（TestConfig）

    Note:
        参数验证已在 TestConfig._validate() 中完成
    """
    test_set = MarineTestSet(
        config.test_hr_root,
        upscale=config.upscale,
        mean=config.mean,
        std=config.std,
        is_mwd=_is_mwd_param(config.marine_param),
        sample_q=False
    )

    test_loader = data.DataLoader(test_set, 1, shuffle=False, num_workers=4,
                                  drop_last=False, persistent_workers=True, pin_memory=True)
    return test_loader


class TestSet(data.Dataset):
    """
    抽象类，需要子类声明 self.hr_list
    """
    hr_list: list[str]

    def __init__(self, upscale, mean, std):
        self.upscale = upscale
        self.mean = torch.tensor(mean).unsqueeze(-1).unsqueeze(-1)
        self.std = torch.tensor(std).unsqueeze(-1).unsqueeze(-1)
        if not hasattr(self, "hr_list"):
            raise NotImplementedError("子类必须在super()前初始化hr_list")

    def load_file(self, index):
        # shape(H, W)
        hr = np.load(self.hr_list[index])
        hr = torch.tensor(hr, dtype=torch.float32)
        # shape(1, H, W)
        return hr.unsqueeze(0)

    def __len__(self):
        return len(self.hr_list)


class MarineTestSet(TestSet):
    def __init__(self, hr_root, upscale, mean, std, is_mwd=False, sample_q=10):
        self.is_mwd = is_mwd
        hr_list = sorted([os.path.join(hr_root, filename) for filename in os.listdir(hr_root)])
        self.hr_list = hr_list[:sample_q] if sample_q else hr_list
        super(MarineTestSet, self).__init__(upscale, mean, std)

    def __getitem__(self, index):
        hr = self.load_file(index)
        hr = encode_mwd(hr) if self.is_mwd else hr
        hr, lr = resize(hr, self.upscale)
        lr = lr.clip(min=-1, max=1) if self.is_mwd else lr
        lr = normalize(self.mean, self.std, lr)
        _, filename = os.path.split(self.hr_list[index])
        return lr.contiguous(), hr.contiguous(), filename


class TrainSet(TestSet):
    def __init__(self, upscale, lr_patch_size=32, mean=0, std=1):
        self.hr_patch_size = lr_patch_size * upscale
        self.resize_fn = transforms.Resize((lr_patch_size, lr_patch_size), transforms.InterpolationMode.BICUBIC)
        super(TrainSet, self).__init__(upscale, mean, std)

    def resize(self, hr, scale):
        lr = self.resize_fn(hr)
        return lr

    def get_hr_patch(self, hr):
        patch_size = self.hr_patch_size
        x0 = random.randint(0, hr.shape[-2] - patch_size)
        y0 = random.randint(0, hr.shape[-1] - patch_size)
        crop_hr = hr[:, x0: x0 + patch_size, y0: y0 + patch_size]
        return crop_hr

    def augment(self, lr, hr):
        hflip = True and random.random() < 0.5
        vflip = True and random.random() < 0.5
        rot = True and random.random() < 0.5

        def augment_(x):
            if hflip:
                x = x.flip(-2)
            if vflip:
                x = x.flip(-1)
            if rot:
                x = x.transpose(-2, -1)
            return x

        lr = augment_(lr)
        hr = augment_(hr)
        return lr, hr


class MarineTrainSet(TestSet):
    def __init__(self, hr_root, upscale, lr_patch_size, mean, std, is_mwd=False):
        self.is_mwd = is_mwd
        self.hr_list = sorted([os.path.join(hr_root, filename) for filename in os.listdir(hr_root)])
        self.hr_patch_size = int(lr_patch_size * upscale)
        self.resize_fn = transforms.Resize((lr_patch_size, lr_patch_size), transforms.InterpolationMode.BICUBIC)
        super(MarineTrainSet, self).__init__(upscale, mean, std)

    def __getitem__(self, index):
        hr = self.load_file(index)
        hr = patchify(hr, self.hr_patch_size)
        # mwd特殊处理
        hr = encode_mwd(hr) if self.is_mwd else hr
        lr = self.resize_fn(hr)
        # mwd特殊处理
        lr = lr.clip(min=-1, max=1) if self.is_mwd else lr
        lr, hr = augement(lr, hr)
        lr, hr = normalize(self.mean, self.std, lr, hr)
        return lr.contiguous(), hr.contiguous()


class UniformTrainSet(data.Dataset):
    def __init__(self, hr_paths, upscale, lr_patch_size, mean: list, std: list):
        super(UniformTrainSet, self).__init__()
        self.hr_patch_size = int(lr_patch_size * upscale)
        self.resize_fn = transforms.Resize((lr_patch_size, lr_patch_size), transforms.InterpolationMode.BICUBIC)
        self.mean = mean  # [[wind, ] [mwd_cos, mwd_sin], [mwp,], [swh,]]
        self.std = std
        hr_files = []  # [[wind1, wind2, ...], [mwd1, mwd2, ...], [...], [...]]
        for single_param_path in hr_paths:
            files = [os.path.join(single_param_path, f) for f in os.listdir(single_param_path)]
            hr_files.append(files)
        # 把（类别编号，文件路径）作为一个元组存入列表
        # self.samples = [(0, wind1), (0, wind2), ..., (3, swh1), (3, swh2)]
        self.samples = []
        for marine_param_idx, files in enumerate(hr_files):
            for file_path in files:
                self.samples.append((marine_param_idx, file_path))

    def normlization(self, img, marine_param_idx=None):
        mean = self.mean[marine_param_idx]
        std = self.std[marine_param_idx]
        # shape: (C, 1, 1)
        mean = torch.tensor(mean).unsqueeze(-1).unsqueeze(-1)
        std = torch.tensor(std).unsqueeze(-1).unsqueeze(-1)
        return (img - mean) / std

    def load_file(self, file_path):
        hr = np.load(file_path)
        hr = torch.tensor(hr, dtype=torch.float32).unsqueeze(0)  # hr: (1, H, W)
        return hr

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        marine_param_idx, file_path = self.samples[index]
        hr = self.load_file(file_path)
        hr = patchify(hr, self.hr_patch_size)
        # 如果取到 mwd 数据
        hr = encode_mwd(hr) if marine_param_idx == 1 else hr
        lr = self.resize_fn(hr)
        lr = lr.clip(min=-1, max=1) if marine_param_idx == 1 else lr
        lr, hr = augement(lr, hr)
        lr, hr = self.normlization(lr, marine_param_idx), self.normlization(hr, marine_param_idx)
        return lr.contiguous(), hr.contiguous(), marine_param_idx


class UniformClassBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size):
        """
        dataset: UniformTrainSet
        batch_size: 每个批次的样本数
        """
        # 按类别分组样本索引
        self.class_indices = {}

        # 遍历dataset.samples中的(param_idx, file_path)对
        for idx, sample in enumerate(dataset.samples):
            param_idx = sample[0]  # 获取类别索引
            if param_idx not in self.class_indices:
                self.class_indices[param_idx] = []
            self.class_indices[param_idx].append(idx)

        self.batch_size = batch_size
        self.dataset_length = len(dataset)

        # 计算每个类别的样本数
        self.class_counts = {k: len(v) for k, v in self.class_indices.items()}
        print(f"类别分布: {self.class_counts}")

    def __iter__(self):
        # 打乱每个类别内的样本顺序
        for param_idx in self.class_indices:
            random.shuffle(self.class_indices[param_idx])

        # 为每个类别创建批次
        all_batches = []
        for param_idx in self.class_indices:
            indices = self.class_indices[param_idx]
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size:  # 只保留完整批次
                    all_batches.append(batch)

        # 打乱所有批次的顺序（保持批次内部类别一致）
        random.shuffle(all_batches)
        return iter(all_batches)

    def __len__(self):
        # 计算完整批次的总数
        return sum(len(indices) // self.batch_size for indices in self.class_indices.values())
