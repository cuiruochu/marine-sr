"""
Single-parameter datasets for SISR (Single Image Super-Resolution)

处理单个海洋参数的超分辨率数据集。
"""

import os
import torch
import numpy as np
import torch.utils.data as data
from torchvision import transforms
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
    val_loader = data.DataLoader(
        val_set, 1, shuffle=False, num_workers=4,
        drop_last=False, persistent_workers=True, pin_memory=True
    )
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
    test_loader = data.DataLoader(
        test_set, 1, shuffle=False, num_workers=4,
        drop_last=False, persistent_workers=True, pin_memory=True
    )
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
        hr = np.load(self.hr_list[index])
        hr = torch.tensor(hr, dtype=torch.float32)
        return hr.unsqueeze(0)

    def __len__(self):
        return len(self.hr_list)


class MarineTestSet(TestSet):
    """海洋参数测试数据集"""

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


class MarineTrainSet(TestSet):
    """海洋参数训练数据集"""

    def __init__(self, hr_root, upscale, lr_patch_size, mean, std, is_mwd=False):
        self.is_mwd = is_mwd
        self.hr_list = sorted([os.path.join(hr_root, filename) for filename in os.listdir(hr_root)])
        self.hr_patch_size = int(lr_patch_size * upscale)
        self.resize_fn = transforms.Resize((lr_patch_size, lr_patch_size), transforms.InterpolationMode.BICUBIC)
        super(MarineTrainSet, self).__init__(upscale, mean, std)

    def __getitem__(self, index):
        hr = self.load_file(index)
        hr = patchify(hr, self.hr_patch_size)
        # MWD 特殊处理
        hr = encode_mwd(hr) if self.is_mwd else hr
        lr = self.resize_fn(hr)
        lr = lr.clip(min=-1, max=1) if self.is_mwd else lr
        lr, hr = augement(lr, hr)
        lr, hr = normalize(self.mean, self.std, lr, hr)
        return lr.contiguous(), hr.contiguous()