"""运行时配置解析。"""

from __future__ import annotations

from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf

from src.app.config_types import (
    CheckpointConfig,
    EvaluateAppConfig,
    EvaluateConfig,
    EvaluateDatasetConfig,
    InferAppConfig,
    InferConfig,
    InferDatasetConfig,
    LossConfig,
    ModelConfig,
    NormalizeConfig,
    OptimizerConfig,
    ResumeConfig,
    SchedulerConfig,
    TrainAppConfig,
    TrainConfig,
    TrainDatasetConfig,
    TrainPathsConfig,
    WandbConfig,
)
from src.app.config_validation import validate_evaluate_config, validate_infer_config, validate_train_config


def load_train_config(raw_cfg: DictConfig | Mapping[str, Any]) -> TrainAppConfig:
    cfg_dict = _to_dict(raw_cfg)
    cfg = TrainAppConfig(
        seed=int(cfg_dict.get("seed", 0)),
        models=_parse_models_config(cfg_dict),
        data_norm=_parse_data_norm_config(cfg_dict),
        upscale=int(cfg_dict["upscale"]),
        dataset=_parse_train_dataset_config(cfg_dict),
        train=TrainConfig(
            epochs=int(cfg_dict["train"]["epochs"]),
            batch_size=int(cfg_dict["train"]["batch_size"]),
            num_workers=int(cfg_dict["train"].get("num_workers", 8)),
            val_num_workers=int(cfg_dict["train"].get("val_num_workers", 4)),
            lr=float(cfg_dict["train"]["lr"]),
            optimizer=OptimizerConfig(
                name=str(cfg_dict["train"]["optimizer"]["name"]),
                params=dict(cfg_dict["train"]["optimizer"].get("params", {})),
            ),
            scheduler=SchedulerConfig(
                name=str(cfg_dict["train"]["scheduler"]["name"]),
                params=dict(cfg_dict["train"]["scheduler"].get("params", {})),
            ),
            loss=LossConfig(
                name=str(cfg_dict["train"]["loss"]["name"]),
                params=dict(cfg_dict["train"]["loss"].get("params", {})),
            ),
            checkpoint=CheckpointConfig(
                every=int(cfg_dict["train"]["checkpoint"]["every"]),
                save_best=bool(cfg_dict["train"]["checkpoint"]["save_best"]),
                monitor=str(cfg_dict["train"]["checkpoint"]["monitor"]),
                mode=str(cfg_dict["train"]["checkpoint"]["mode"]),
            ),
        ),
        resume=ResumeConfig(
            checkpoint=normalize_optional_string(cfg_dict.get("resume", {}).get("checkpoint")),
            load_optimizer=bool(cfg_dict.get("resume", {}).get("load_optimizer", True)),
            load_scheduler=bool(cfg_dict.get("resume", {}).get("load_scheduler", True)),
            load_callbacks=bool(cfg_dict.get("resume", {}).get("load_callbacks", True)),
            load_rng_state=bool(cfg_dict.get("resume", {}).get("load_rng_state", True)),
        ),
        paths=TrainPathsConfig(
            checkpoint_dir=str(cfg_dict["paths"]["checkpoint_dir"]),
            experiment_dir=str(cfg_dict["paths"]["experiment_dir"]),
        ),
        wandb=WandbConfig(
            project=str(cfg_dict["wandb"]["project"]),
            mode=str(cfg_dict["wandb"]["mode"]),
        ),
    )
    validate_train_config(cfg)
    return cfg


def load_evaluate_config(raw_cfg: DictConfig | Mapping[str, Any]) -> EvaluateAppConfig:
    cfg_dict = _to_dict(raw_cfg)
    cfg = EvaluateAppConfig(
        models=_parse_models_config(cfg_dict),
        data_norm=_parse_data_norm_config(cfg_dict),
        upscale=int(cfg_dict["upscale"]),
        dataset=_parse_evaluate_dataset_config(cfg_dict),
        evaluate=EvaluateConfig(
            checkpoint=normalize_optional_string(cfg_dict["evaluate"].get("checkpoint")),
            save_results=bool(cfg_dict["evaluate"]["save_results"]),
            save_dir=str(cfg_dict["evaluate"]["save_dir"]),
            mask=normalize_optional_string(cfg_dict["evaluate"].get("mask")),
            batch_size=int(cfg_dict["evaluate"]["batch_size"]),
            num_workers=int(cfg_dict["evaluate"]["num_workers"]),
        ),
    )
    validate_evaluate_config(cfg)
    return cfg


def load_infer_config(raw_cfg: DictConfig | Mapping[str, Any]) -> InferAppConfig:
    cfg_dict = _to_dict(raw_cfg)
    cfg = InferAppConfig(
        models=_parse_models_config(cfg_dict),
        data_norm=_parse_data_norm_config(cfg_dict),
        upscale=int(cfg_dict["upscale"]),
        dataset=_parse_infer_dataset_config(cfg_dict),
        infer=InferConfig(
            checkpoint=normalize_optional_string(cfg_dict["infer"].get("checkpoint")),
            save_results=bool(cfg_dict["infer"]["save_results"]),
            save_dir=str(cfg_dict["infer"]["save_dir"]),
            mask=normalize_optional_string(cfg_dict["infer"].get("mask")),
            batch_size=int(cfg_dict["infer"]["batch_size"]),
            num_workers=int(cfg_dict["infer"]["num_workers"]),
        ),
    )
    validate_infer_config(cfg)
    return cfg


def load_eval_task_config(raw_cfg: DictConfig | Mapping[str, Any]) -> EvaluateAppConfig | InferAppConfig:
    cfg_dict = _to_dict(raw_cfg)
    if "evaluate" in cfg_dict:
        return load_evaluate_config(cfg_dict)
    if "infer" in cfg_dict:
        return load_infer_config(cfg_dict)
    raise ValueError("评估类配置必须包含 'evaluate' 或 'infer' 段")


def normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return str(value)


def parse_max_sample(value: Any) -> int | bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return int(value)


def _parse_models_config(cfg_dict: Mapping[str, Any]) -> ModelConfig:
    section = cfg_dict.get("models")
    if section is None:
        raise KeyError("缺少 models 配置段")
    return ModelConfig(
        name=str(section["name"]),
        in_dim=int(section["in_dim"]) if section.get("in_dim") is not None else None,
        params=dict(section.get("params", {})),
    )


def _parse_train_dataset_config(cfg_dict: Mapping[str, Any]) -> TrainDatasetConfig:
    dataset = cfg_dict["dataset"]
    return TrainDatasetConfig(
        name=str(dataset["name"]),
        lr_patch_size=int(dataset["lr_patch_size"]),
        train_lr_root=str(dataset["train_lr_root"]),
        train_hr_root=str(dataset["train_hr_root"]),
        val_lr_root=str(dataset["val_lr_root"]),
        val_hr_root=str(dataset["val_hr_root"]),
        max_sample=parse_max_sample(dataset.get("max_sample", False)),
    )


def _parse_evaluate_dataset_config(cfg_dict: Mapping[str, Any]) -> EvaluateDatasetConfig:
    dataset = cfg_dict["dataset"]
    return EvaluateDatasetConfig(
        name=str(dataset["name"]),
        eval_lr_root=str(dataset["eval_lr_root"]),
        eval_hr_root=str(dataset["eval_hr_root"]),
        max_sample=parse_max_sample(dataset.get("max_sample", False)),
    )


def _parse_infer_dataset_config(cfg_dict: Mapping[str, Any]) -> InferDatasetConfig:
    dataset = cfg_dict["dataset"]
    return InferDatasetConfig(
        name=str(dataset["name"]),
        infer_lr_root=str(dataset["infer_lr_root"]),
    )


def _parse_data_norm_config(cfg_dict: Mapping[str, Any]) -> NormalizeConfig:
    section = cfg_dict["data_norm"]
    return NormalizeConfig(
        mean=[float(x) for x in section["mean"]],
        std=[float(x) for x in section["std"]],
    )


def _to_dict(raw_cfg: DictConfig | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(raw_cfg, DictConfig):
        return OmegaConf.to_container(raw_cfg, resolve=True)
    return dict(raw_cfg)
