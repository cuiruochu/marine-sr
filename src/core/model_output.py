"""
统一模型输出协议

将不同模型的前向返回值归一化为同一结构，避免训练/评估引擎感知具体模型细节。
"""

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class ModelOutput:
    pred: torch.Tensor
    aux_losses: dict[str, torch.Tensor] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


def normalize_model_output(output: Any) -> ModelOutput:
    if isinstance(output, ModelOutput):
        return ModelOutput(
            pred=_require_tensor(output.pred, "pred"),
            aux_losses=_normalize_aux_losses(output.aux_losses),
            extras=dict(output.extras),
        )

    if torch.is_tensor(output):
        return ModelOutput(pred=output)

    if isinstance(output, (tuple, list)):
        if not output:
            raise TypeError("模型输出不能为空 tuple/list")

        pred = _require_tensor(output[0], "pred")
        if len(output) == 1:
            return ModelOutput(pred=pred)
        if len(output) > 2:
            raise TypeError("模型 tuple/list 输出最多支持两个元素: (pred, aux)")

        aux = output[1]
        if torch.is_tensor(aux):
            return ModelOutput(pred=pred, aux_losses=_normalize_aux_losses({"aux_loss": aux}))
        if isinstance(aux, Mapping):
            aux_dict = dict(aux)
            aux_losses = aux_dict.pop("aux_losses", {})
            if "aux_loss" in aux_dict:
                aux_losses = {"aux_loss": aux_dict.pop("aux_loss"), **dict(aux_losses)}
            return ModelOutput(
                pred=pred,
                aux_losses=_normalize_aux_losses(aux_losses),
                extras=aux_dict,
            )

        raise TypeError("模型辅助输出必须是 Tensor 或 Mapping")

    raise TypeError(f"不支持的模型输出类型: {type(output)!r}")


def sum_aux_losses(aux_losses: Mapping[str, torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
    if not aux_losses:
        return torch.zeros((), device=reference.device, dtype=reference.dtype)

    total = torch.zeros((), device=reference.device, dtype=reference.dtype)
    for value in aux_losses.values():
        total = total + value.to(device=reference.device, dtype=reference.dtype)
    return total


def _normalize_aux_losses(aux_losses: Mapping[str, torch.Tensor] | None) -> dict[str, torch.Tensor]:
    if not aux_losses:
        return {}

    normalized = {}
    for name, value in aux_losses.items():
        normalized[name] = _require_scalar_tensor(value, f"aux_losses.{name}")
    return normalized


def _require_tensor(value: Any, field_name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{field_name} 必须是 torch.Tensor")
    return value


def _require_scalar_tensor(value: Any, field_name: str) -> torch.Tensor:
    tensor = _require_tensor(value, field_name)
    if tensor.numel() != 1:
        raise ValueError(f"{field_name} 必须是标量 Tensor，当前 shape={tuple(tensor.shape)}")
    return tensor.reshape(())
