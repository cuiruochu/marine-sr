"""运行时配置校验。"""

from __future__ import annotations

from pathlib import Path

from src.app.config_types import EvaluateAppConfig, InferAppConfig, TrainAppConfig
from src.losses.registry import list_losses
from src.models.registry import list_models
from src.optim.optimizer import list_optimizers
from src.optim.scheduler import list_schedulers


def validate_train_config(cfg: TrainAppConfig) -> None:
    _validate_model_name(cfg.models.name)
    _validate_model_in_dim(cfg.models.in_dim)
    _validate_data_norm("data_norm", cfg.data_norm)
    _validate_dataset_common("dataset", cfg.dataset.name)
    _validate_positive_int("upscale", cfg.upscale)
    _validate_positive_int("dataset.lr_patch_size", cfg.dataset.lr_patch_size)
    _validate_non_empty("dataset.train_lr_root", cfg.dataset.train_lr_root)
    _validate_non_empty("dataset.train_hr_root", cfg.dataset.train_hr_root)
    _validate_non_empty("dataset.val_lr_root", cfg.dataset.val_lr_root)
    _validate_non_empty("dataset.val_hr_root", cfg.dataset.val_hr_root)
    _validate_distinct_paths("dataset.train_lr_root", cfg.dataset.train_lr_root, "dataset.train_hr_root", cfg.dataset.train_hr_root)
    _validate_distinct_paths("dataset.val_lr_root", cfg.dataset.val_lr_root, "dataset.val_hr_root", cfg.dataset.val_hr_root)
    _validate_max_sample("dataset.max_sample", cfg.dataset.max_sample)

    _validate_choice("train.optimizer.name", cfg.train.optimizer.name, list_optimizers())
    _validate_scheduler_name(cfg.train.scheduler.name)
    _validate_choice("train.loss.name", cfg.train.loss.name, list_losses())
    _validate_choice("train.checkpoint.mode", cfg.train.checkpoint.mode, ["min", "max"])
    _validate_choice("wandb.mode", cfg.wandb.mode, ["offline", "online", "disabled"])

    _validate_positive_int("seed", cfg.seed, allow_zero=True)
    _validate_positive_int("train.epochs", cfg.train.epochs)
    _validate_positive_int("train.batch_size", cfg.train.batch_size)
    _validate_positive_int("train.num_workers", cfg.train.num_workers, allow_zero=True)
    _validate_positive_int("train.val_num_workers", cfg.train.val_num_workers, allow_zero=True)
    _validate_positive_float("train.lr", cfg.train.lr)
    _validate_positive_int("train.checkpoint.every", cfg.train.checkpoint.every)
    if cfg.train.checkpoint.every > cfg.train.epochs:
        raise ValueError("train.checkpoint.every 不能大于 train.epochs")
    _validate_non_empty("train.checkpoint.monitor", cfg.train.checkpoint.monitor)
    if cfg.resume.checkpoint is not None:
        _validate_checkpoint_suffix("resume.checkpoint", cfg.resume.checkpoint)

    _validate_non_empty("paths.checkpoint_dir", cfg.paths.checkpoint_dir)
    _validate_non_empty("paths.experiment_dir", cfg.paths.experiment_dir)
    _validate_non_empty("wandb.project", cfg.wandb.project)


def validate_evaluate_config(cfg: EvaluateAppConfig) -> None:
    _validate_model_name(cfg.models.name)
    _validate_model_in_dim(cfg.models.in_dim)
    _validate_data_norm("data_norm", cfg.data_norm)
    _validate_dataset_common("dataset", cfg.dataset.name)
    _validate_positive_int("upscale", cfg.upscale)
    _validate_non_empty("dataset.eval_lr_root", cfg.dataset.eval_lr_root)
    _validate_non_empty("dataset.eval_hr_root", cfg.dataset.eval_hr_root)
    _validate_distinct_paths("dataset.eval_lr_root", cfg.dataset.eval_lr_root, "dataset.eval_hr_root", cfg.dataset.eval_hr_root)
    _validate_max_sample("dataset.max_sample", cfg.dataset.max_sample)
    _validate_positive_int("evaluate.batch_size", cfg.evaluate.batch_size)
    _validate_positive_int("evaluate.num_workers", cfg.evaluate.num_workers, allow_zero=True)

    if cfg.evaluate.checkpoint is not None:
        _validate_checkpoint_suffix("evaluate.checkpoint", cfg.evaluate.checkpoint)
    if cfg.evaluate.mask is not None:
        _validate_npy_suffix("evaluate.mask", cfg.evaluate.mask)
    if cfg.evaluate.save_results:
        _validate_non_empty("evaluate.save_dir", cfg.evaluate.save_dir)


def validate_infer_config(cfg: InferAppConfig) -> None:
    _validate_model_name(cfg.models.name)
    _validate_model_in_dim(cfg.models.in_dim)
    _validate_data_norm("data_norm", cfg.data_norm)
    _validate_dataset_common("dataset", cfg.dataset.name)
    _validate_positive_int("upscale", cfg.upscale)
    _validate_non_empty("dataset.infer_lr_root", cfg.dataset.infer_lr_root)
    _validate_positive_int("infer.batch_size", cfg.infer.batch_size)
    _validate_positive_int("infer.num_workers", cfg.infer.num_workers, allow_zero=True)

    if cfg.infer.checkpoint is not None:
        _validate_checkpoint_suffix("infer.checkpoint", cfg.infer.checkpoint)
    if cfg.infer.mask is not None:
        _validate_npy_suffix("infer.mask", cfg.infer.mask)
    if cfg.infer.save_results:
        _validate_non_empty("infer.save_dir", cfg.infer.save_dir)


def _validate_model_name(name: str) -> None:
    _validate_choice("models.name", name, list_models())


def _validate_model_in_dim(value: int | None) -> None:
    if value is None:
        return
    _validate_positive_int("models.in_dim", value)


def _validate_dataset_common(field_prefix: str, name: str) -> None:
    _validate_non_empty(f"{field_prefix}.name", name)


def _validate_data_norm(field_prefix: str, normalize) -> None:
    if len(normalize.mean) != len(normalize.std):
        raise ValueError(f"{field_prefix}.mean 与 {field_prefix}.std 长度必须一致")
    if len(normalize.mean) == 0:
        raise ValueError(f"{field_prefix}.mean 不能为空")
    if any(value <= 0 for value in normalize.std):
        raise ValueError(f"{field_prefix}.std 中的值必须大于 0")


def _validate_choice(field_name: str, value: str, choices: list[str]) -> None:
    if value not in choices:
        raise ValueError(f"{field_name} 无效: '{value}'。可用选项: {choices}")


def _validate_scheduler_name(name: str) -> None:
    if not name or name.lower() == "none":
        return
    _validate_choice("train.scheduler.name", name, list_schedulers())


def _validate_non_empty(field_name: str, value: str) -> None:
    if not str(value).strip():
        raise ValueError(f"{field_name} 不能为空")


def _validate_positive_int(field_name: str, value: int, allow_zero: bool = False) -> None:
    if allow_zero:
        if value < 0:
            raise ValueError(f"{field_name} 必须大于等于 0")
        return
    if value <= 0:
        raise ValueError(f"{field_name} 必须大于 0")


def _validate_positive_float(field_name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{field_name} 必须大于 0")


def _validate_distinct_paths(field_a: str, path_a: str, field_b: str, path_b: str) -> None:
    if Path(path_a).resolve() == Path(path_b).resolve():
        raise ValueError(f"{field_a} 与 {field_b} 不能指向同一路径")


def _validate_checkpoint_suffix(field_name: str, value: str) -> None:
    _validate_non_empty(field_name, value)
    if Path(value).suffix.lower() not in {".pth", ".pt"}:
        raise ValueError(f"{field_name} 必须以 .pth 或 .pt 结尾")


def _validate_npy_suffix(field_name: str, value: str) -> None:
    _validate_non_empty(field_name, value)
    if Path(value).suffix.lower() != ".npy":
        raise ValueError(f"{field_name} 必须是 .npy 文件")


def _validate_max_sample(field_name: str, value: int | bool) -> None:
    if value is False:
        return
    if value is True:
        raise ValueError(f"{field_name} 不能为 true；请使用正整数或 false")
    if int(value) <= 0:
        raise ValueError(f"{field_name} 必须为正整数或 false")
