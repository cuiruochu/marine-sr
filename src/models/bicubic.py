"""Bicubic 基线模型。"""

from __future__ import annotations

import torch
from torch import nn

from .base import create_model_result
from .registry import register_model


class BicubicUpsampler(nn.Module):
    def __init__(self, upscale: int):
        super().__init__()
        self.upscale = int(upscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Bicubic 仅支持 NCHW 输入，当前 shape={tuple(x.shape)}")

        try:
            from torchvision.transforms import InterpolationMode
            from torchvision.transforms.functional import resize
        except ImportError as exc:
            raise ModuleNotFoundError("Bicubic 模型依赖 torchvision，请先安装 torchvision。") from exc

        height, width = x.shape[-2:]
        return resize(
            x,
            size=[height * self.upscale, width * self.upscale],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )


@register_model("bicubic", default_params={})
def create_bicubic(params: dict, in_dim: int, upscale: int):
    del params, in_dim
    return create_model_result(BicubicUpsampler(upscale=upscale), "bicubic")
