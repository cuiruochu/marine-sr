"""
运行时配置模型

按任务拆分为训练、评估、推理三类配置，只共享模型配置。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf

from src.losses import list_losses
from src.models import list_models
from src.optim import list_optimizers, list_schedulers
from src.utils import PROJECT_ROOT


@dataclass(frozen=True)
class NormalizeConfig:
    mean: list[float]
    std: list[float]


@dataclass(frozen=True)
class ModelConfig:
    name: str
    params: dict[str, Any]


@dataclass(frozen=True)
class TrainDatasetConfig:
    name: str
    upscale: int
    lr_patch_size: int
    channels: int
    normalize: NormalizeConfig
    train_lr_root: str
    train_hr_root: str
    val_lr_root: str
    val_hr_root: str
    max_sample: int | bool


@dataclass(frozen=True)
class EvaluateDatasetConfig:
    name: str
    upscale: int
    channels: int
    normalize: NormalizeConfig
    eval_lr_root: str
    eval_hr_root: str
    max_sample: int | bool


@dataclass(frozen=True)
class InferDatasetConfig:
    name: str
    upscale: int
    channels: int
    normalize: NormalizeConfig
    infer_lr_root: str


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    params: dict[str, Any]


@dataclass(frozen=True)
class SchedulerConfig:
    name: str
    params: dict[str, Any]


@dataclass(frozen=True)
class LossConfig:
    name: str
    params: dict[str, Any]


@dataclass(frozen=True)
class CheckpointConfig:
    every: int
    save_best: bool
    monitor: str
    mode: str


@dataclass(frozen=True)
class ResumeConfig:
    checkpoint: str | None
    load_optimizer: bool
    load_scheduler: bool
    load_rng_state: bool


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    batch_size: int
    num_workers: int
    val_num_workers: int
    lr: float
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    loss: LossConfig
    checkpoint: CheckpointConfig


@dataclass(frozen=True)
class EvaluateConfig:
    checkpoint: str | None
    save_results: bool
    save_dir: str
    eval_mask: str | None
    batch_size: int
    num_workers: int


@dataclass(frozen=True)
class InferConfig:
    checkpoint: str | None
    save_results: bool
    save_dir: str
    batch_size: int
    num_workers: int


@dataclass(frozen=True)
class TrainPathsConfig:
    checkpoint_dir: str
    experiment_dir: str


@dataclass(frozen=True)
class WandbConfig:
    project: str
    mode: str


@dataclass(frozen=True)
class ModelBuildSpec:
    model_name: str
    model_params: dict[str, Any]
    in_dim: int
    upscale: int


@dataclass(frozen=True)
class TrainLoaderSpec:
    upscale: int
    lr_patch_size: int
    mean: list[float]
    std: list[float]
    train_lr_root: str
    train_hr_root: str
    val_lr_root: str
    val_hr_root: str
    max_sample: int | bool


@dataclass(frozen=True)
class TestLoaderSpec:
    upscale: int
    mean: list[float]
    std: list[float]
    lr_root: str
    hr_root: str | None
    mode: str
    max_sample: int | bool


class _ModelBuildMixin:
    model: ModelConfig
    dataset: Any

    @property
    def model_build_spec(self) -> ModelBuildSpec:
        return ModelBuildSpec(
            model_name=self.model.name,
            model_params=self.model.params,
            in_dim=self.dataset.channels,
            upscale=self.dataset.upscale,
        )


@dataclass(frozen=True)
class TrainAppConfig(_ModelBuildMixin):
    seed: int
    model: ModelConfig
    dataset: TrainDatasetConfig
    train: TrainConfig
    resume: ResumeConfig
    paths: TrainPathsConfig
    wandb: WandbConfig

    @property
    def train_loader_spec(self) -> TrainLoaderSpec:
        return TrainLoaderSpec(
            upscale=self.dataset.upscale,
            lr_patch_size=self.dataset.lr_patch_size,
            mean=self.dataset.normalize.mean,
            std=self.dataset.normalize.std,
            train_lr_root=str(_resolve_project_path(self.dataset.train_lr_root)),
            train_hr_root=str(_resolve_project_path(self.dataset.train_hr_root)),
            val_lr_root=str(_resolve_project_path(self.dataset.val_lr_root)),
            val_hr_root=str(_resolve_project_path(self.dataset.val_hr_root)),
            max_sample=self.dataset.max_sample,
        )

    @property
    def checkpoint_root(self) -> Path:
        return PROJECT_ROOT / self.paths.checkpoint_dir / self.model.name / self.dataset.name / f"x{self.dataset.upscale}"


class _EvalTaskMixin(_ModelBuildMixin):
    @property
    def results_root(self) -> Path:
        return Path(self.save_dir) / self.model.name / self.dataset.name / f"x{self.dataset.upscale}"


@dataclass(frozen=True)
class EvaluateAppConfig(_EvalTaskMixin):
    model: ModelConfig
    dataset: EvaluateDatasetConfig
    evaluate: EvaluateConfig

    @property
    def mode(self) -> str:
        return "evaluation"

    @property
    def checkpoint(self) -> str | None:
        return self.evaluate.checkpoint

    @property
    def save_results(self) -> bool:
        return self.evaluate.save_results

    @property
    def save_dir(self) -> str:
        return self.evaluate.save_dir

    @property
    def eval_mask(self) -> str | None:
        return self.evaluate.eval_mask

    @property
    def batch_size(self) -> int:
        return self.evaluate.batch_size

    @property
    def num_workers(self) -> int:
        return self.evaluate.num_workers

    @property
    def test_loader_spec(self) -> TestLoaderSpec:
        return TestLoaderSpec(
            upscale=self.dataset.upscale,
            mean=self.dataset.normalize.mean,
            std=self.dataset.normalize.std,
            lr_root=str(_resolve_project_path(self.dataset.eval_lr_root)),
            hr_root=str(_resolve_project_path(self.dataset.eval_hr_root)),
            mode=self.mode,
            max_sample=self.dataset.max_sample,
        )


@dataclass(frozen=True)
class InferAppConfig(_EvalTaskMixin):
    model: ModelConfig
    dataset: InferDatasetConfig
    infer: InferConfig

    @property
    def mode(self) -> str:
        return "inference"

    @property
    def checkpoint(self) -> str | None:
        return self.infer.checkpoint

    @property
    def save_results(self) -> bool:
        return self.infer.save_results

    @property
    def save_dir(self) -> str:
        return self.infer.save_dir

    @property
    def eval_mask(self) -> None:
        return None

    @property
    def batch_size(self) -> int:
        return self.infer.batch_size

    @property
    def num_workers(self) -> int:
        return self.infer.num_workers

    @property
    def test_loader_spec(self) -> TestLoaderSpec:
        return TestLoaderSpec(
            upscale=self.dataset.upscale,
            mean=self.dataset.normalize.mean,
            std=self.dataset.normalize.std,
            lr_root=str(_resolve_project_path(self.dataset.infer_lr_root)),
            hr_root=None,
            mode=self.mode,
            max_sample=False,
        )


def load_train_config(raw_cfg: DictConfig | Mapping[str, Any]) -> TrainAppConfig:
    cfg_dict = _to_dict(raw_cfg)
    cfg = TrainAppConfig(
        seed=int(cfg_dict["seed"]),
        model=_parse_model_config(cfg_dict),
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
            checkpoint=cfg_dict.get("resume", {}).get("checkpoint"),
            load_optimizer=bool(cfg_dict.get("resume", {}).get("load_optimizer", True)),
            load_scheduler=bool(cfg_dict.get("resume", {}).get("load_scheduler", True)),
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
    _validate_train_config(cfg)
    return cfg


def load_evaluate_config(raw_cfg: DictConfig | Mapping[str, Any]) -> EvaluateAppConfig:
    cfg_dict = _to_dict(raw_cfg)
    cfg = EvaluateAppConfig(
        model=_parse_model_config(cfg_dict),
        dataset=_parse_evaluate_dataset_config(cfg_dict),
        evaluate=EvaluateConfig(
            checkpoint=cfg_dict["evaluate"].get("checkpoint"),
            save_results=bool(cfg_dict["evaluate"]["save_results"]),
            save_dir=str(cfg_dict["evaluate"]["save_dir"]),
            eval_mask=_normalize_optional_string(cfg_dict["evaluate"].get("eval_mask")),
            batch_size=int(cfg_dict["evaluate"]["batch_size"]),
            num_workers=int(cfg_dict["evaluate"]["num_workers"]),
        ),
    )
    _validate_evaluate_config(cfg)
    return cfg


def load_infer_config(raw_cfg: DictConfig | Mapping[str, Any]) -> InferAppConfig:
    cfg_dict = _to_dict(raw_cfg)
    cfg = InferAppConfig(
        model=_parse_model_config(cfg_dict),
        dataset=_parse_infer_dataset_config(cfg_dict),
        infer=InferConfig(
            checkpoint=cfg_dict["infer"].get("checkpoint"),
            save_results=bool(cfg_dict["infer"]["save_results"]),
            save_dir=str(cfg_dict["infer"]["save_dir"]),
            batch_size=int(cfg_dict["infer"]["batch_size"]),
            num_workers=int(cfg_dict["infer"]["num_workers"]),
        ),
    )
    _validate_infer_config(cfg)
    return cfg


def load_eval_task_config(raw_cfg: DictConfig | Mapping[str, Any]) -> EvaluateAppConfig | InferAppConfig:
    cfg_dict = _to_dict(raw_cfg)
    if "evaluate" in cfg_dict:
        return load_evaluate_config(cfg_dict)
    if "infer" in cfg_dict:
        return load_infer_config(cfg_dict)
    raise ValueError("评估类配置必须包含 'evaluate' 或 'infer' 段")


def _parse_model_config(cfg_dict: Mapping[str, Any]) -> ModelConfig:
    section = cfg_dict.get("models")
    if section is None:
        raise KeyError("缺少 models 配置段")
    return ModelConfig(
        name=str(section["name"]),
        params=dict(section.get("params", {})),
    )


def _parse_normalize_config(cfg_dict: Mapping[str, Any]) -> NormalizeConfig:
    return NormalizeConfig(
        mean=[float(x) for x in cfg_dict["normalize"]["mean"]],
        std=[float(x) for x in cfg_dict["normalize"]["std"]],
    )


def _parse_train_dataset_config(cfg_dict: Mapping[str, Any]) -> TrainDatasetConfig:
    dataset = cfg_dict["dataset"]
    return TrainDatasetConfig(
        name=str(dataset["name"]),
        upscale=int(dataset["upscale"]),
        lr_patch_size=int(dataset["lr_patch_size"]),
        channels=int(dataset["channels"]),
        normalize=_parse_normalize_config(dataset),
        train_lr_root=str(dataset["train_lr_root"]),
        train_hr_root=str(dataset["train_hr_root"]),
        val_lr_root=str(dataset["val_lr_root"]),
        val_hr_root=str(dataset["val_hr_root"]),
        max_sample=_parse_max_sample(dataset.get("max_sample", False)),
    )


def _parse_evaluate_dataset_config(cfg_dict: Mapping[str, Any]) -> EvaluateDatasetConfig:
    dataset = cfg_dict["dataset"]
    return EvaluateDatasetConfig(
        name=str(dataset["name"]),
        upscale=int(dataset["upscale"]),
        channels=int(dataset["channels"]),
        normalize=_parse_normalize_config(dataset),
        eval_lr_root=str(dataset["eval_lr_root"]),
        eval_hr_root=str(dataset["eval_hr_root"]),
        max_sample=_parse_max_sample(dataset.get("max_sample", False)),
    )


def _parse_infer_dataset_config(cfg_dict: Mapping[str, Any]) -> InferDatasetConfig:
    dataset = cfg_dict["dataset"]
    return InferDatasetConfig(
        name=str(dataset["name"]),
        upscale=int(dataset["upscale"]),
        channels=int(dataset["channels"]),
        normalize=_parse_normalize_config(dataset),
        infer_lr_root=str(dataset["infer_lr_root"]),
    )


def _to_dict(raw_cfg: DictConfig | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(raw_cfg, DictConfig):
        return OmegaConf.to_container(raw_cfg, resolve=True)
    return dict(raw_cfg)


def _validate_model_config(model: ModelConfig) -> None:
    _validate_choice("model.name", model.name, list_models())


def _validate_dataset_common(field_prefix: str, dataset: Any) -> None:
    _validate_non_empty(f"{field_prefix}.name", dataset.name)
    _validate_positive_int(f"{field_prefix}.upscale", dataset.upscale)
    _validate_positive_int(f"{field_prefix}.channels", dataset.channels)

    if len(dataset.normalize.mean) != dataset.channels:
        raise ValueError(
            f"{field_prefix}.normalize.mean 长度必须等于 {field_prefix}.channels。当前 channels={dataset.channels}, "
            f"mean 长度={len(dataset.normalize.mean)}"
        )
    if len(dataset.normalize.std) != dataset.channels:
        raise ValueError(
            f"{field_prefix}.normalize.std 长度必须等于 {field_prefix}.channels。当前 channels={dataset.channels}, "
            f"std 长度={len(dataset.normalize.std)}"
        )
    if any(value <= 0 for value in dataset.normalize.std):
        raise ValueError(f"{field_prefix}.normalize.std 中的值必须大于 0")


def _validate_train_config(cfg: TrainAppConfig) -> None:
    _validate_model_config(cfg.model)
    _validate_dataset_common("dataset", cfg.dataset)
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
    if cfg.resume.checkpoint is not None:
        _validate_checkpoint_suffix("resume.checkpoint", cfg.resume.checkpoint)

    _validate_non_empty("paths.checkpoint_dir", cfg.paths.checkpoint_dir)
    _validate_non_empty("paths.experiment_dir", cfg.paths.experiment_dir)
    _validate_non_empty("wandb.project", cfg.wandb.project)
    _validate_non_empty("train.checkpoint.monitor", cfg.train.checkpoint.monitor)


def _validate_evaluate_config(cfg: EvaluateAppConfig) -> None:
    _validate_model_config(cfg.model)
    _validate_dataset_common("dataset", cfg.dataset)
    _validate_non_empty("dataset.eval_lr_root", cfg.dataset.eval_lr_root)
    _validate_non_empty("dataset.eval_hr_root", cfg.dataset.eval_hr_root)
    _validate_distinct_paths("dataset.eval_lr_root", cfg.dataset.eval_lr_root, "dataset.eval_hr_root", cfg.dataset.eval_hr_root)
    _validate_max_sample("dataset.max_sample", cfg.dataset.max_sample)
    _validate_positive_int("evaluate.batch_size", cfg.evaluate.batch_size)
    _validate_positive_int("evaluate.num_workers", cfg.evaluate.num_workers, allow_zero=True)

    if cfg.evaluate.checkpoint is not None:
        _validate_checkpoint_suffix("evaluate.checkpoint", cfg.evaluate.checkpoint)
    if cfg.evaluate.eval_mask is not None:
        _validate_npy_suffix("evaluate.eval_mask", cfg.evaluate.eval_mask)
    if cfg.evaluate.save_results:
        _validate_non_empty("evaluate.save_dir", cfg.evaluate.save_dir)


def _validate_infer_config(cfg: InferAppConfig) -> None:
    _validate_model_config(cfg.model)
    _validate_dataset_common("dataset", cfg.dataset)
    _validate_non_empty("dataset.infer_lr_root", cfg.dataset.infer_lr_root)
    _validate_positive_int("infer.batch_size", cfg.infer.batch_size)
    _validate_positive_int("infer.num_workers", cfg.infer.num_workers, allow_zero=True)

    if cfg.infer.checkpoint is not None:
        _validate_checkpoint_suffix("infer.checkpoint", cfg.infer.checkpoint)
    if cfg.infer.save_results:
        _validate_non_empty("infer.save_dir", cfg.infer.save_dir)


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
    if _resolve_project_path(path_a) == _resolve_project_path(path_b):
        raise ValueError(f"{field_a} 与 {field_b} 不能指向同一路径")


def _validate_checkpoint_suffix(field_name: str, value: str) -> None:
    _validate_non_empty(field_name, value)
    suffix = Path(value).suffix.lower()
    if suffix not in {".pth", ".pt"}:
        raise ValueError(f"{field_name} 必须以 .pth 或 .pt 结尾")


def _validate_npy_suffix(field_name: str, value: str) -> None:
    _validate_non_empty(field_name, value)
    if Path(value).suffix.lower() != ".npy":
        raise ValueError(f"{field_name} 必须是 .npy 文件")


def _normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return str(value)


def _parse_max_sample(value: Any) -> int | bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return int(value)


def _validate_max_sample(field_name: str, value: int | bool) -> None:
    if value is False:
        return
    if value is True:
        raise ValueError(f"{field_name} 不能为 true；请使用正整数或 false")
    if int(value) <= 0:
        raise ValueError(f"{field_name} 必须为正整数或 false")


def _resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
