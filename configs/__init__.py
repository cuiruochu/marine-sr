import torch
import random
import numpy as np

from .loader import TrainConfig, TestConfig, load_yaml


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_train_config(yaml_path: str = "configs/train.yaml") -> TrainConfig:
    """获取训练配置"""
    config = TrainConfig.from_yaml(yaml_path)
    print(f"当前训练 {config.marine_param} single 模型!!!")
    return config


def get_test_config(yaml_path: str = "configs/test.yaml") -> TestConfig:
    """获取测试配置"""
    return TestConfig.from_yaml(yaml_path)


__all__ = [
    "setup_seed",
    "load_yaml",
    "TrainConfig",
    "TestConfig",
    "get_train_config",
    "get_test_config",
]