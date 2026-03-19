"""
训练引擎

提供训练循环、评估循环和回调机制的统一入口。
支持单卡训练和分布式训练（DDP）。
"""

from typing import Callable, List, Optional, Dict, Any, Union
import torch
import numpy as np
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from .callbacks import Callback, CallbackList
from .metrics import (
    calculate_psnr,
    calculate_ssim,
    calculate_mae,
    calculate_max_mae,
    reverse_norm,
    normalize_to_01,
    apply_mask,
)
from .model_output import normalize_model_output, sum_aux_losses
from src.utils import capture_rng_state, restore_rng_state
from src.utils.distributed import (
    is_distributed,
    get_rank,
    get_world_size,
    is_main_process,
    barrier,
)


class Engine:
    """
    训练引擎

    职责：
    - 训练循环 (train_step, fit)
    - 评估循环 (evaluate)
    - 回调触发 (callbacks)
    - 状态管理 (current_epoch, global_step)
    - 分布式训练支持 (DDP)

    用法：
        engine = Engine(
            model=model,
            optimizer=optimizer,
            loss_fn=nn.L1Loss(),
            callbacks=[CheckpointCallback(...), WandbCallback(...)]
        )
        engine.fit(train_loader, val_loader, epochs=200, mean=[0], std=[1])

    分布式训练：
        engine = Engine(
            model=model,
            optimizer=optimizer,
            loss_fn=nn.L1Loss(),
            ddp=True,  # 启用 DDP
        )
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: Callable,
        scheduler: Optional[LRScheduler] = None,
        device: Optional[str] = None,
        callbacks: Optional[List[Callback]] = None,
        ddp: bool = False,
        config_snapshot: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化训练引擎

        Args:
            model: 模型
            optimizer: 优化器
            loss_fn: 损失函数
            scheduler: 学习率调度器（可选）
            device: 设备，默认自动检测
            callbacks: 回调列表
            ddp: 是否启用分布式数据并行（DDP）
        """
        # 设备
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 核心组件
        self.model = model.to(self.device)
        self._base_model = self.model  # 保存原始模型引用

        # DDP 包装
        self.ddp = ddp or is_distributed()
        if self.ddp and is_distributed():
            # 获取 local_rank 用于 device_ids
            local_rank = get_rank()  # 在 torchrun 下，每个进程只有一个 GPU
            self.model = DDP(self.model, device_ids=[self.device], output_device=self.device)

        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.config_snapshot = config_snapshot

        # 回调
        self.callbacks = CallbackList(callbacks or [])

        # 状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float("inf")
        self.best_epoch = 0

        # 配置
        self._train_loader = None
        self._val_loader = None

    @property
    def lr(self) -> float:
        """当前学习率"""
        return self.optimizer.param_groups[0]["lr"]

    def train_step(self, batch: Any) -> Dict[str, Any]:
        """
        单个训练步骤

        Args:
            batch: 数据批次 (lr, hr, ...)

        Returns:
            指标字典 {"loss": float}
        """
        self.model.train()

        # 解析数据
        lr, hr = batch[0].to(self.device), batch[1].to(self.device)

        # 前向传播
        self.optimizer.zero_grad()
        output = normalize_model_output(self.model(lr))
        sr = output.pred
        recon_loss = self.loss_fn(sr, hr)
        aux_loss = sum_aux_losses(output.aux_losses, recon_loss)
        loss = recon_loss + aux_loss

        # 反向传播
        loss.backward()
        self.optimizer.step()

        logs = {
            "loss": loss.item(),
            "recon_loss": recon_loss.item(),
        }
        if output.aux_losses:
            logs["aux_loss"] = aux_loss.item()
            for name, value in output.aux_losses.items():
                logs[f"aux_{name}"] = value.detach().item()
        return logs

    @torch.no_grad()
    def evaluate(
        self,
        val_loader: Any,
        mean: List[float],
        std: List[float],
        mask: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        评估

        Args:
            val_loader: 验证数据加载器
            mean: 归一化均值
            std: 归一化标准差
            mask: 评估掩码（可选）

        Returns:
            指标字典 {"val_psnr", "val_ssim", "val_mae", "val_max_mae"}
        """
        self.model.eval()

        # 准备归一化参数
        mean_t = torch.tensor(mean, device=self.device).view(-1, 1, 1)
        std_t = torch.tensor(std, device=self.device).view(-1, 1, 1)

        # 准备掩码
        mask_t = None
        if mask is not None:
            mask_t = torch.from_numpy(mask).to(self.device).unsqueeze(0).unsqueeze(0)

        # 累计指标
        total_psnr, total_ssim, total_mae, total_max_mae = 0.0, 0.0, 0.0, 0.0
        num_batches = 0

        for batch in val_loader:
            lr, hr = batch[0].to(self.device), batch[1].to(self.device)
            sr = normalize_model_output(self.model(lr)).pred

            # 反归一化
            sr = reverse_norm(sr, mean_t, std_t)

            # 应用掩码
            if mask_t is not None:
                hr, sr = apply_mask(mask_t.expand_as(hr), hr, sr)

            # 归一化到 [0, 1] 用于 PSNR/SSIM
            hr_norm, sr_norm = normalize_to_01(hr, sr)

            # 计算指标
            total_mae += calculate_mae(sr, hr, mask=mask_t)
            total_max_mae = max(total_max_mae, calculate_max_mae(sr, hr, mask=mask_t))
            total_psnr += calculate_psnr(hr_norm, sr_norm, mask=mask_t)
            total_ssim += calculate_ssim(hr_norm, sr_norm, mask=mask_t)
            num_batches += 1

        return {
            "val_psnr": total_psnr / num_batches,
            "val_ssim": total_ssim / num_batches,
            "val_mae": total_mae / num_batches,
            "val_max_mae": total_max_mae,
        }

    def fit(
        self,
        train_loader: Any,
        val_loader: Any,
        epochs: int,
        mean: List[float],
        std: List[float],
        mask: Optional[np.ndarray] = None,
        start_epoch: int = 1,
    ) -> Dict[str, Any]:
        """
        主训练循环

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            mean: 归一化均值
            std: 归一化标准差
            mask: 评估掩码（可选）
            start_epoch: 起始 epoch（用于恢复训练）

        Returns:
            训练历史
        """
        self._train_loader = train_loader
        self._val_loader = val_loader

        # 触发训练开始回调
        self.callbacks.on_train_begin(self)

        history = {"train_loss": [], "val_psnr": [], "val_ssim": [], "val_mae": []}

        for epoch in range(start_epoch, start_epoch + epochs):
            self.current_epoch = epoch

            # 触发 epoch 开始回调
            self.callbacks.on_epoch_begin(self, epoch)

            # 训练一个 epoch
            epoch_loss = self._train_epoch(train_loader)

            # 评估
            val_logs = self.evaluate(val_loader, mean, std, mask)
            val_logs["lr"] = self.lr
            val_logs["train_loss"] = epoch_loss

            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step()

            # 触发 epoch 结束回调
            self.callbacks.on_epoch_end(self, epoch, val_logs)

            # 记录历史
            history["train_loss"].append(epoch_loss)
            history["val_psnr"].append(val_logs["val_psnr"])
            history["val_ssim"].append(val_logs["val_ssim"])
            history["val_mae"].append(val_logs["val_mae"])

        # 触发训练结束回调
        self.callbacks.on_train_end(self)

        return history

    def _train_epoch(self, train_loader: Any) -> float:
        """
        训练一个 epoch

        Args:
            train_loader: 训练数据加载器

        Returns:
            平均损失
        """
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}", leave=False)

        for batch_idx, batch in enumerate(pbar, 1):
            # 触发 batch 开始回调
            self.callbacks.on_batch_begin(self, batch_idx)

            # 训练步骤
            logs = self.train_step(batch)
            self.global_step += 1

            # 触发 batch 结束回调
            self.callbacks.on_batch_end(self, batch_idx, logs)

            # 更新进度条
            pbar.set_postfix(loss=f"{logs['loss']:.4f}", lr=f"{self.lr:.2e}")

            total_loss += logs["loss"]
            num_batches += 1

        return total_loss / num_batches

    def get_model(self) -> nn.Module:
        """
        获取原始模型（用于评估和保存）

        在 DDP 模式下，返回被包装的原始模型。

        Returns:
            原始模型实例
        """
        if self.ddp and hasattr(self.model, 'module'):
            return self.model.module
        return self.model

    def save_checkpoint(self, path: str, epoch: Optional[int] = None) -> None:
        """
        保存检查点

        在 DDP 模式下，只保存原始模型的权重，不保存 DDP wrapper。
        只在主进程执行保存操作。

        Args:
            path: 保存路径
            epoch: epoch 数，默认使用当前 epoch
        """
        # 只在主进程保存
        if not is_main_process():
            return

        epoch = epoch or self.current_epoch

        # 获取原始模型（不包含 DDP wrapper）
        model = self.get_model()

        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "model": model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "callbacks": self.callbacks.state_dict(),
            "rng_state": capture_rng_state(),
        }

        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
        if self.config_snapshot is not None:
            state["config"] = self.config_snapshot

        torch.save(state, path)

    def load_checkpoint(
        self,
        path: str,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        load_rng_state: bool = True,
    ) -> int:
        """
        加载检查点

        Args:
            path: 检查点路径
            load_optimizer: 是否加载优化器状态
            load_scheduler: 是否加载调度器状态

        Returns:
            保存的 epoch 数
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=True)

        # 加载到原始模型（不包含 DDP wrapper）
        model = self.get_model()
        model.load_state_dict(ckpt["model"])

        if load_optimizer and "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])

        if (
            load_scheduler
            and self.scheduler is not None
            and "scheduler" in ckpt
        ):
            self.scheduler.load_state_dict(ckpt["scheduler"])

        # 恢复状态
        self.current_epoch = ckpt.get("epoch", 0)
        self.global_step = ckpt.get("global_step", 0)
        self.best_metric = ckpt.get("best_metric", float("inf"))
        self.best_epoch = ckpt.get("best_epoch", 0)
        self.callbacks.load_state_dict(ckpt.get("callbacks"))
        if load_rng_state:
            restore_rng_state(ckpt.get("rng_state"))

        return self.current_epoch

    def count_parameters(self, trainable_only: bool = True) -> int:
        """
        统计参数数量

        Args:
            trainable_only: 是否只统计可训练参数

        Returns:
            参数数量
        """
        model = self.get_model()
        if trainable_only:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        return sum(p.numel() for p in model.parameters())
