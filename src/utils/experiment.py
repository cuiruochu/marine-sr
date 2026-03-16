"""
实验管理模块

提供实验目录创建、配置保存等功能。
"""

import os
import json
import yaml
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional

from src.utils.path import PROJECT_ROOT


class ExperimentManager:
    """
    实验管理器

    功能：
    - 创建实验目录
    - 保存配置副本
    - 管理 checkpoint
    - 记录训练日志
    """

    def __init__(self, config, base_dir: str = "experiments"):
        """
        Args:
            config: TrainConfig 实例
            base_dir: 实验根目录
        """
        self.config = config
        self.base_dir = Path(base_dir)
        if not self.base_dir.is_absolute():
            self.base_dir = PROJECT_ROOT / self.base_dir

        # 生成实验名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_name = f"{timestamp}_{config.model_name}_{config.marine_param}_x{config.upscale}"

        # 实验目录
        self.exp_dir = self.base_dir / self.exp_name
        self.checkpoint_dir = self.exp_dir / "checkpoints"
        self.log_file = self.exp_dir / "train_log.json"

        # 日志数据
        self.logs = {
            "config": config.raw,
            "metrics": []
        }

    def setup(self) -> str:
        """
        创建实验目录结构

        Returns:
            实验目录路径
        """
        # 创建目录
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 保存配置副本
        self._save_config()

        # 初始化日志文件
        self._init_log()

        return str(self.exp_dir)

    def _save_config(self):
        """保存配置副本到实验目录"""
        config_path = self.exp_dir / "config.yaml"

        # 保存原始配置
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(self.config.raw, f, default_flow_style=False, allow_unicode=True)

    def _init_log(self):
        """初始化日志文件"""
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(self.logs, f, indent=2, ensure_ascii=False)

    def log_metrics(self, epoch: int, metrics: dict):
        """
        记录训练指标

        Args:
            epoch: 当前 epoch
            metrics: 指标字典，如 {"loss": 0.1, "psnr": 30.5, "lr": 2e-4}
        """
        entry = {"epoch": epoch}
        entry.update(metrics)
        self.logs["metrics"].append(entry)

        # 写入文件
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(self.logs, f, indent=2, ensure_ascii=False)

    def get_checkpoint_path(self, epoch: int) -> str:
        """
        获取 checkpoint 保存路径

        Args:
            epoch: epoch 数

        Returns:
            checkpoint 文件路径
        """
        return str(self.checkpoint_dir / f"epoch_{epoch}.pth")

    def get_best_checkpoint_path(self) -> str:
        """获取最佳模型保存路径"""
        return str(self.checkpoint_dir / "best.pth")

    def save_checkpoint(self, state: dict, epoch: int, is_best: bool = False):
        """
        保存 checkpoint

        Args:
            state: 模型状态字典
            epoch: 当前 epoch
            is_best: 是否为最佳模型
        """
        # 保存常规 checkpoint
        path = self.get_checkpoint_path(epoch)
        import torch
        torch.save(state, path)

        # 如果是最佳模型，额外保存一份
        if is_best:
            best_path = self.get_best_checkpoint_path()
            shutil.copy(path, best_path)

    def get_wandb_name(self) -> str:
        """获取 wandb 实验名（与目录名一致）"""
        return self.exp_name

    def __repr__(self):
        return f"ExperimentManager({self.exp_name})"


def create_experiment(config, base_dir: str = "experiments") -> ExperimentManager:
    """
    创建实验管理器

    Args:
        config: TrainConfig 实例
        base_dir: 实验根目录

    Returns:
        ExperimentManager 实例
    """
    manager = ExperimentManager(config, base_dir)
    manager.setup()
    return manager