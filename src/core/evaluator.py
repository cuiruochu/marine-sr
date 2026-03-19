"""
评估引擎

提供评估循环和回调机制的统一入口。
"""

from typing import List, Optional, Dict, Any
import torch
import numpy as np
from torch import nn
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
from .model_output import normalize_model_output


class Evaluator:
    """
    评估引擎

    职责：
    - 模型加载 (load_checkpoint)
    - 评估循环 (run)
    - 单步评估 (eval_step)
    - 回调触发 (callbacks)

    用法：
        evaluator = Evaluator(
            model=model,
            callbacks=[MetricsCallback(), SaveResultsCallback(...)]
        )
        evaluator.load_checkpoint(checkpoint_path)
        metrics = evaluator.run(test_loader, mean, std, mask)

    回调触发顺序：
        on_eval_begin()
        for batch in test_loader:
            on_batch_begin()
            sr, hr, filename = eval_step()
            on_batch_end(sr, hr, filename)
        on_eval_end(metrics)
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        callbacks: Optional[List[Callback]] = None,
    ):
        """
        初始化评估引擎

        Args:
            model: 模型
            device: 设备，默认自动检测
            callbacks: 回调列表
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.callbacks = CallbackList(callbacks or [])

        # 状态
        self.current_batch = 0
        self.total_batches = 0

    def load_checkpoint(self, path: str) -> None:
        """
        加载模型权重

        Args:
            path: 检查点路径
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=True)

        # 支持完整检查点或仅权重
        if "model" in ckpt:
            self.model.load_state_dict(ckpt["model"])
        else:
            self.model.load_state_dict(ckpt)

        self.model.eval()

    @torch.no_grad()
    def run(
        self,
        test_loader: Any,
        mean: List[float],
        std: List[float],
        mask: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        运行评估

        Args:
            test_loader: 测试数据加载器
            mean: 归一化均值
            std: 归一化标准差
            mask: 评估掩码（可选）

        Returns:
            evaluation 模式返回指标字典，inference 模式返回空字典
        """
        self.model.eval()
        self.total_batches = len(test_loader)
        self.current_batch = 0

        # 准备归一化参数
        mean_t = torch.tensor(mean, device=self.device).view(-1, 1, 1)
        std_t = torch.tensor(std, device=self.device).view(-1, 1, 1)

        # 准备掩码
        mask_t = None
        if mask is not None:
            mask_t = torch.from_numpy(mask).to(self.device).unsqueeze(0).unsqueeze(0)
            # 转换为 bool 类型用于索引
            if mask_t.dtype != torch.bool:
                mask_t = mask_t > 0.5

        # 触发评估开始回调
        self.callbacks.on_eval_begin(self)

        # 累计指标
        total_psnr, total_ssim, total_mae, total_max_mae = 0.0, 0.0, 0.0, 0.0
        metric_batches = 0

        pbar = tqdm(test_loader, desc="Evaluating")

        for batch in pbar:
            self.current_batch += 1

            # 触发 batch 开始回调
            self.callbacks.on_batch_begin(self, self.current_batch)

            # 评估单步
            sr, hr, filename = self._eval_step(
                batch, mean_t, std_t, mask_t
            )

            if hr is None:
                self.callbacks.on_batch_end(
                    self,
                    self.current_batch,
                    sr=sr,
                    hr=None,
                    filename=filename,
                    metrics=None,
                )
                continue

            # 计算指标
            hr_masked, sr_masked = hr, sr
            if mask_t is not None:
                hr_masked, sr_masked = apply_mask(mask_t.expand_as(hr), hr, sr)

            hr_norm, sr_norm = normalize_to_01(hr_masked, sr_masked)

            batch_psnr = calculate_psnr(hr_norm, sr_norm, mask=mask_t)
            batch_ssim = calculate_ssim(hr_norm, sr_norm, mask=mask_t)
            batch_mae = calculate_mae(sr_masked, hr_masked, mask=mask_t)
            batch_max_mae = calculate_max_mae(sr_masked, hr_masked, mask=mask_t)

            total_psnr += batch_psnr
            total_ssim += batch_ssim
            total_mae += batch_mae
            total_max_mae = max(total_max_mae, batch_max_mae)
            metric_batches += 1

            # 触发 batch 结束回调
            self.callbacks.on_batch_end(
                self, self.current_batch,
                sr=sr, hr=hr, filename=filename,
                metrics={"psnr": batch_psnr, "ssim": batch_ssim, "mae": batch_mae}
            )

            # 更新进度条
            pbar.set_postfix(
                psnr=f"{batch_psnr:.2f}",
                ssim=f"{batch_ssim:.4f}",
                mae=f"{batch_mae:.4f}"
            )

        # 计算平均指标
        num_batches = len(test_loader)
        if metric_batches == 0:
            metrics = {}
        else:
            metrics = {
                "psnr": total_psnr / metric_batches,
                "ssim": total_ssim / metric_batches,
                "mae": total_mae / metric_batches,
                "max_mae": total_max_mae,
            }

        # 触发评估结束回调
        self.callbacks.on_eval_end(self, metrics)

        return metrics

    def _eval_step(
        self,
        batch: Any,
        mean_t: torch.Tensor,
        std_t: torch.Tensor,
        mask_t: Optional[torch.Tensor],
    ) -> tuple:
        """
        单步评估

        Args:
            batch: 数据批次，evaluation 为 (lr, hr, filename)，inference 为 (lr, filename)
            mean_t: 归一化均值张量
            std_t: 归一化标准差张量
            mask_t: 掩码张量

        Returns:
            (sr, hr, filename) - SR结果、HR真值或 None、文件名
        """
        if len(batch) == 3:
            lr, hr, filename = batch[0].to(self.device), batch[1].to(self.device), batch[2]
        elif len(batch) == 2:
            lr, filename = batch[0].to(self.device), batch[1]
            hr = None
        else:
            raise ValueError(f"不支持的评估 batch 结构，长度={len(batch)}")

        # 推理
        sr = normalize_model_output(self.model(lr)).pred

        # 反归一化
        sr = reverse_norm(sr, mean_t, std_t)

        return sr, hr, filename[0] if isinstance(filename, tuple) else filename

    def count_parameters(self, trainable_only: bool = True) -> int:
        """
        统计参数数量

        Args:
            trainable_only: 是否只统计可训练参数

        Returns:
            参数数量
        """
        if trainable_only:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.model.parameters())
