"""
模型输出规范化
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch


@dataclass
class ModelOutput:
    """标准化的模型输出"""

    pred: torch.Tensor
    aux_losses: Optional[Dict[str, torch.Tensor]] = None


def normalize_model_output(output: Union[torch.Tensor, Tuple, ModelOutput]) -> ModelOutput:
    """
    规范化模型输出为统一的 ModelOutput 格式。

    支持的输出形式：
    - 直接返回 pred 张量
    - 返回 (pred, aux_loss)
    - 返回 (pred, {"aux_losses": {...}})
    - 返回 ModelOutput
    """
    if isinstance(output, ModelOutput):
        return output

    if isinstance(output, torch.Tensor):
        return ModelOutput(pred=output)

    if isinstance(output, (tuple, list)):
        pred = output[0]
        aux_losses = None

        if len(output) >= 2:
            second = output[1]
            if isinstance(second, dict):
                aux_losses = second.get("aux_losses", {})
            elif isinstance(second, torch.Tensor):
                aux_losses = {"aux": second}

        return ModelOutput(pred=pred, aux_losses=aux_losses)

    raise ValueError(f"不支持的模型输出类型: {type(output)}")


def sum_aux_losses(aux_losses: Optional[Dict[str, torch.Tensor]], base_loss: torch.Tensor) -> torch.Tensor:
    """汇总辅助损失。"""
    if not aux_losses:
        return torch.tensor(0.0, device=base_loss.device)

    total = torch.tensor(0.0, device=base_loss.device)
    for loss in aux_losses.values():
        total = total + loss
    return total
