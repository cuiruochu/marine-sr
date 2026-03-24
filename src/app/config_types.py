"""运行时配置类型。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.path import PROJECT_ROOT, resolve_project_path


@dataclass(frozen=True)
class NormalizeConfig:
    mean: list[float]
    std: list[float]


@dataclass(frozen=True)
class DatasetPairConfig:
    lr_root: str
    hr_root: str
    normalize: NormalizeConfig
    max_sample: int | bool = False


@dataclass(frozen=True)
class ModelConfig:
    name: str
    in_dim: int | None
    params: dict[str, Any]


@dataclass(frozen=True)
class TrainDatasetConfig:
    name: str
    lr_patch_size: int
    train_lr_root: str
    train_hr_root: str
    val_lr_root: str
    val_hr_root: str
    max_sample: int | bool = False


@dataclass(frozen=True)
class EvaluateDatasetConfig:
    name: str
    eval_lr_root: str
    eval_hr_root: str
    max_sample: int | bool = False


@dataclass(frozen=True)
class InferDatasetConfig:
    name: str
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
    load_callbacks: bool
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
    mask: str | None
    batch_size: int
    num_workers: int


@dataclass(frozen=True)
class InferConfig:
    checkpoint: str | None
    save_results: bool
    save_dir: str
    mask: str | None
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
    train_pair: DatasetPairConfig
    val_pair: DatasetPairConfig


@dataclass(frozen=True)
class EvaluationLoaderSpec:
    upscale: int
    normalize: NormalizeConfig
    eval_pair: DatasetPairConfig


@dataclass(frozen=True)
class InferenceLoaderSpec:
    upscale: int
    normalize: NormalizeConfig
    infer_lr_root: str


TestLoaderSpec = EvaluationLoaderSpec | InferenceLoaderSpec


class _ModelBuildMixin:
    models: ModelConfig
    dataset: Any
    data_norm: NormalizeConfig
    upscale: int

    @property
    def models_build_spec(self) -> ModelBuildSpec:
        return ModelBuildSpec(
            model_name=self.models.name,
            model_params=self.models.params,
            in_dim=self.models.in_dim if self.models.in_dim is not None else len(self.data_norm.mean),
            upscale=self.upscale,
        )


@dataclass(frozen=True)
class TrainAppConfig(_ModelBuildMixin):
    seed: int
    models: ModelConfig
    data_norm: NormalizeConfig
    upscale: int
    dataset: TrainDatasetConfig
    train: TrainConfig
    resume: ResumeConfig
    paths: TrainPathsConfig
    wandb: WandbConfig

    @property
    def train_loader_spec(self) -> TrainLoaderSpec:
        train_pair = DatasetPairConfig(
            lr_root=str(resolve_project_path(self.dataset.train_lr_root)),
            hr_root=str(resolve_project_path(self.dataset.train_hr_root)),
            normalize=self.data_norm,
        )
        val_pair = DatasetPairConfig(
            lr_root=str(resolve_project_path(self.dataset.val_lr_root)),
            hr_root=str(resolve_project_path(self.dataset.val_hr_root)),
            normalize=self.data_norm,
            max_sample=self.dataset.max_sample,
        )
        return TrainLoaderSpec(
            upscale=self.upscale,
            lr_patch_size=self.dataset.lr_patch_size,
            train_pair=train_pair,
            val_pair=val_pair,
        )

    @property
    def checkpoint_root(self) -> Path:
        return PROJECT_ROOT / self.paths.checkpoint_dir / self.models.name / self.dataset.name / f"x{self.upscale}"


class _EvalTaskMixin(_ModelBuildMixin):
    @property
    def results_root(self) -> Path:
        return Path(self.save_dir) / self.models.name / self.dataset.name / f"x{self.upscale}"


@dataclass(frozen=True)
class EvaluateAppConfig(_EvalTaskMixin):
    models: ModelConfig
    data_norm: NormalizeConfig
    upscale: int
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
    def mask(self) -> str | None:
        return self.evaluate.mask

    @property
    def batch_size(self) -> int:
        return self.evaluate.batch_size

    @property
    def num_workers(self) -> int:
        return self.evaluate.num_workers

    @property
    def test_loader_spec(self) -> EvaluationLoaderSpec:
        pair = DatasetPairConfig(
            lr_root=str(resolve_project_path(self.dataset.eval_lr_root)),
            hr_root=str(resolve_project_path(self.dataset.eval_hr_root)),
            normalize=self.data_norm,
            max_sample=self.dataset.max_sample,
        )
        return EvaluationLoaderSpec(
            upscale=self.upscale,
            normalize=self.data_norm,
            eval_pair=pair,
        )


@dataclass(frozen=True)
class InferAppConfig(_EvalTaskMixin):
    models: ModelConfig
    data_norm: NormalizeConfig
    upscale: int
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
    def mask(self) -> str | None:
        return self.infer.mask

    @property
    def batch_size(self) -> int:
        return self.infer.batch_size

    @property
    def num_workers(self) -> int:
        return self.infer.num_workers

    @property
    def test_loader_spec(self) -> InferenceLoaderSpec:
        return InferenceLoaderSpec(
            upscale=self.upscale,
            normalize=self.data_norm,
            infer_lr_root=str(resolve_project_path(self.dataset.infer_lr_root)),
        )


EvalTaskConfig = EvaluateAppConfig | InferAppConfig
