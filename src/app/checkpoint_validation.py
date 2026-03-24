"""Checkpoint 与当前任务配置的一致性校验。"""

from pathlib import Path
from typing import Any, Mapping

import torch

from src.app.config_types import EvalTaskConfig, TrainAppConfig
from src.models.registry import normalize_model_name


def validate_checkpoint_matches_config(
    cfg: TrainAppConfig | EvalTaskConfig,
    checkpoint_path: str | Path,
) -> None:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"检查点格式无效: {checkpoint_path}")

    signature = _extract_checkpoint_signature(checkpoint.get("config"))
    if signature is None:
        return

    current = _build_current_signature(cfg)
    mismatches = []
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

    model_section = config.get("models")
    dataset_section = config.get("dataset")
    if not isinstance(model_section, Mapping) or not isinstance(dataset_section, Mapping):
        return None

    signature = {
        "models.name": normalize_model_name(model_section.get("name")),
        "models.in_dim": model_section.get("in_dim"),
        "models.params": dict(model_section.get("params", {})),
        "upscale": config.get("upscale", dataset_section.get("upscale")),
    }
    return signature


def _build_current_signature(cfg: TrainAppConfig | EvalTaskConfig) -> dict[str, Any]:
    return {
        "models.name": normalize_model_name(cfg.models.name),
        "models.in_dim": cfg.models.in_dim,
        "models.params": dict(cfg.models.params),
        "upscale": cfg.upscale,
    }
