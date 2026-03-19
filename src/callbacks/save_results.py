"""
保存结果回调

将 SR 图像保存为 .npy 文件。
"""

from typing import Any, Dict, Optional
from pathlib import Path
import os
import torch
import numpy as np

from src.core.callbacks import Callback
from src.core.metrics import reverse_norm, apply_mask


class SaveResultsCallback(Callback):
    """
    保存结果回调

    将 SR 图像保存为 .npy 文件。

    用法：
        evaluator = Evaluator(
            model=model,
            callbacks=[SaveResultsCallback(save_dir="./results")]
        )
    """

    def __init__(
        self,
        save_dir: str,
        save_format: str = "npy",
        mean: Optional[list] = None,
        std: Optional[list] = None,
        mask: Optional[np.ndarray] = None,
    ):
        """
        初始化

        Args:
            save_dir: 保存目录
            save_format: 保存格式，目前仅支持 "npy"
            mean: 归一化均值（如果需要再次反归一化）
            std: 归一化标准差
            mask: 应用掩码（可选）
        """
        self.save_dir = Path(save_dir)
        self.save_format = save_format
        self.mean = mean
        self.std = std
        self.mask = mask

    def on_eval_begin(self, evaluator: Any) -> None:
        """创建保存目录"""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = evaluator.device

        # 准备掩码张量
        self.mask_t = None
        if self.mask is not None:
            self.mask_t = torch.from_numpy(self.mask).to(self.device).unsqueeze(0).unsqueeze(0)

    def on_batch_end(
        self,
        evaluator: Any,
        batch_idx: int,
        logs: Optional[Dict[str, Any]] = None,
        sr: torch.Tensor = None,
        hr: torch.Tensor = None,
        filename: str = None,
        **kwargs,
    ) -> None:
        """保存 SR 结果"""
        if sr is None or filename is None:
            return

        # 应用掩码（如果有）
        if self.mask_t is not None:
            sr = apply_mask(self.mask_t.expand_as(sr), sr)

        # 移除 batch 维度: (1, C, H, W) -> (C, H, W)
        sr = sr.squeeze(0)

        # 转换为 numpy
        sr_np = sr.cpu().numpy()

        # 获取文件名（不含扩展名）
        name, _ = os.path.splitext(filename)

        # 保存
        save_path = self.save_dir / f"{name}.{self.save_format}"
        np.save(save_path, sr_np)
