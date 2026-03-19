"""Checkpoint 与当前任务配置的一致性校验。"""

from pathlib import Path
from typing import Any, Mapping

import torch

from src.app.config import EvaluateAppConfig, InferAppConfig, TrainAppConfig


def validate_checkpoint_matches_config(
    cfg: TrainAppConfig | EvaluateAppConfig | InferAppConfig,
    checkpoint_path: str | Path,
) -> None:
    """校验 checkpoint 中的配置快照是否与当前任务配置一致。"""
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"检查点格式无效: {checkpoint_path}")

    signature = _extract_checkpoint_signature(checkpoint.get("config"))
    if signature is None:
        return

    current = _build_current_signature(cfg)
    mismatches: list[str] = []
    for key, current_value in current.items():
        checkpoint_value = signature.get(key)
        if checkpoint_value is None:
            continue
        if checkpoint_value != current_value:
            mismatches.append(f"{key}: checkpoint={checkpoint_value!r}, current={current_value!r}")

    if mismatches:
        raise ValueError("检查点配置与当前配置不一致: " + "; ".join(mismatches))


def _extract_checkpoint_signature(config: Any) -> dict[str, Any] | None:
    if not isinstance(config, Mapping):
        return None

    model_section = config.get("models") or config.get("model")
    dataset_section = config.get("dataset")
    if not isinstance(model_section, Mapping) or not isinstance(dataset_section, Mapping):
        return None

    return {
        "model.name": model_section.get("name"),
        "model.params": dict(model_section.get("params", {})),
        "dataset.name": dataset_section.get("name"),
        "dataset.upscale": dataset_section.get("upscale"),
        "dataset.channels": dataset_section.get("channels"),
    }


def _build_current_signature(cfg: TrainAppConfig | EvaluateAppConfig | InferAppConfig) -> dict[str, Any]:
    return {
        "model.name": cfg.model.name,
        "model.params": dict(cfg.model.params),
        "dataset.name": cfg.dataset.name,
        "dataset.upscale": cfg.dataset.upscale,
        "dataset.channels": cfg.dataset.channels,
    }
