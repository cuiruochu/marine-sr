"""
训练器模块

使用 Engine + Callback 模式实现训练逻辑。
所有组件通过工厂创建，支持配置驱动。
"""

import torch
from torch import nn, optim

from src.core import Engine
from src.callbacks import (
    CheckpointCallback,
    LoggingCallback,
    WandbCallback,
    ProgressCallback,
    LRMonitorCallback,
)
from src.losses import get_loss, list_losses
from src.optim import get_optimizer, get_scheduler, list_optimizers, list_schedulers


class Trainer:
    """
    统一训练器

    使用 Engine + Callback 模式，职责清晰：
    - Engine: 训练循环、评估循环
    - Callback: 日志、保存、WandB 等

    Args:
        model: 模型
        optimizer: 优化器
        loss_fn: 损失函数
        scheduler: 学习率调度器
        device: 设备
        callbacks: 回调列表

    用法：
        trainer = Trainer(model, optimizer, nn.L1Loss())
        trainer.fit(train_loader, val_loader, epochs=200, mean=[0], std=[1])
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module = None,
        scheduler=None,
        device: str = None,
        callbacks: list = None,
    ):
        self.loss_fn = loss_fn or nn.L1Loss()

        # 创建 Engine
        self.engine = Engine(
            model=model,
            optimizer=optimizer,
            loss_fn=self.loss_fn,
            scheduler=scheduler,
            device=device,
            callbacks=callbacks or [],
        )

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int,
        mean: list,
        std: list,
        mask=None,
        start_epoch: int = 1,
    ):
        """
        训练模型

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            mean: 归一化均值
            std: 归一化标准差
            mask: 评估掩码
            start_epoch: 起始 epoch
        """
        return self.engine.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            mean=mean,
            std=std,
            mask=mask,
            start_epoch=start_epoch,
        )

    def load_checkpoint(self, path: str):
        """加载检查点"""
        return self.engine.load_checkpoint(path)

    def save_checkpoint(self, path: str):
        """保存检查点"""
        self.engine.save_checkpoint(path)

    @property
    def model(self):
        return self.engine.model

    @property
    def current_epoch(self):
        return self.engine.current_epoch


def create_trainer(
    model: nn.Module,
    lr: float = 2e-4,
    epochs: int = 200,
    optimizer_name: str = "adam",
    optimizer_params: dict = None,
    scheduler_name: str = "step",
    scheduler_params: dict = None,
    loss_name: str = "l1",
    loss_params: dict = None,
    device: str = None,
    checkpoint_dir: str = None,
    save_every: int = 10,
    wandb_config: dict = None,
    log_file: str = None,
) -> Trainer:
    """
    工厂函数：创建训练器

    所有组件通过配置指定，无需修改代码。

    Args:
        model: 模型
        lr: 学习率
        epochs: 训练轮数
        optimizer_name: 优化器名称 ("adam", "adamw", "sgd", "rmsprop")
        optimizer_params: 优化器参数 (weight_decay, betas 等)
        scheduler_name: 调度器名称 ("step", "cosine", "multistep", "plateau", "none")
        scheduler_params: 调度器参数
        loss_name: 损失函数名称 ("l1", "l2", "mse", "mae")
        loss_params: 损失函数参数
        device: 设备
        checkpoint_dir: 检查点保存目录
        save_every: 保存频率
        wandb_config: WandB 配置
        log_file: 日志文件路径

    Returns:
        Trainer 实例
    """
    optimizer_params = optimizer_params or {}
    scheduler_params = scheduler_params or {}
    loss_params = loss_params or {}

    # 使用工厂创建组件
    optimizer = get_optimizer(optimizer_name, model, lr=lr, **optimizer_params)
    loss_fn = get_loss(loss_name, **loss_params)
    scheduler = get_scheduler(scheduler_name, optimizer, **scheduler_params)

    # 回调
    callbacks = []

    # 检查点回调
    if checkpoint_dir:
        callbacks.append(
            CheckpointCallback(
                save_dir=checkpoint_dir,
                every=save_every,
                save_best=True,
                monitor="val_mae",
                mode="min",
            )
        )

    # 日志回调
    callbacks.append(LoggingCallback(log_file=log_file))

    # WandB 回调
    if wandb_config:
        callbacks.append(
            WandbCallback(
                project=wandb_config.get("project"),
                name=wandb_config.get("name"),
                config=wandb_config.get("config"),
                mode=wandb_config.get("mode", "offline"),
            )
        )

    # 进度条回调
    callbacks.append(ProgressCallback())

    return Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        device=device,
        callbacks=callbacks,
    )


# 向后兼容：保留 CommonTrainer 别名
CommonTrainer = Trainer


def get_trainer(train_cfg, model_dict, lr, epoches, train_loader, val_cfg, val_loader, wandb_log):
    """
    兼容旧接口：创建训练器

    Args:
        train_cfg: 训练配置
        model_dict: 模型字典 {"model": ..., "model_name": ...}
        lr: 学习率
        epoches: 训练轮数
        train_loader: 训练数据加载器
        val_cfg: 验证配置
        val_loader: 验证数据加载器
        wandb_log: WandB run 对象

    Returns:
        Trainer 实例
    """
    model = model_dict["model"]

    # WandB 配置
    wandb_config = None
    if wandb_log is not None:
        wandb_config = {
            "project": "Marine-Parameters-SupervisedSR",
            "name": model_dict.get("model_name"),
            "mode": "offline",
        }

    # 从配置中获取组件参数
    optimizer_name = getattr(train_cfg, "optimizer_name", "adam")
    optimizer_params = getattr(train_cfg, "optimizer_params", {})
    scheduler_name = getattr(train_cfg, "scheduler_name", "step")
    scheduler_params = getattr(train_cfg, "scheduler_params", {})
    loss_name = getattr(train_cfg, "loss_name", "l1")

    # 如果 scheduler_params 中没有 step_size，使用 epoches 的一半
    if scheduler_name == "step" and "step_size" not in scheduler_params:
        scheduler_params = {**scheduler_params, "step_size": epoches // 2}

    # 创建训练器
    trainer = create_trainer(
        model=model,
        lr=lr,
        epochs=epoches,
        optimizer_name=optimizer_name,
        optimizer_params=optimizer_params,
        scheduler_name=scheduler_name,
        scheduler_params=scheduler_params,
        loss_name=loss_name,
        checkpoint_dir=train_cfg.pth_save_path,
        save_every=10,
        wandb_config=wandb_config,
    )

    # 保存额外信息用于 fit
    trainer._train_loader = train_loader
    trainer._val_loader = val_loader
    trainer._epochs = epoches
    trainer._val_cfg = val_cfg
    trainer._model_name = model_dict.get("model_name")

    return trainer