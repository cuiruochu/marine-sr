"""
运行时对象构建器

基于任务专属配置创建训练、评估、推理所需对象。
"""

from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from src.app.config import EvaluateAppConfig, InferAppConfig, TrainAppConfig
from src.callbacks import (
    CheckpointCallback,
    LoggingCallback,
    MetricsCallback,
    ProgressCallback,
    SaveResultsCallback,
    WandbCallback,
)
from src.data import build_test_loader as build_test_data_loader
from src.data import build_train_val_loaders
from src.losses import get_loss
from src.models import create_model
from src.optim import get_optimizer, get_scheduler


def build_model_bundle(cfg):
    return create_model(cfg.model_build_spec)


def build_train_loaders(cfg: TrainAppConfig):
    return build_train_val_loaders(
        cfg.train_loader_spec,
        train_batch_size=cfg.train.batch_size,
        train_num_workers=cfg.train.num_workers,
        eval_num_workers=cfg.train.val_num_workers,
    )


def build_test_loader(cfg: EvaluateAppConfig | InferAppConfig):
    return build_test_data_loader(
        cfg.test_loader_spec,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )


def build_optimizer(cfg: TrainAppConfig, model):
    return get_optimizer(
        cfg.train.optimizer.name,
        model,
        lr=cfg.train.lr,
        **cfg.train.optimizer.params,
    )


def build_scheduler(cfg: TrainAppConfig, optimizer):
    return get_scheduler(
        cfg.train.scheduler.name,
        optimizer,
        **cfg.train.scheduler.params,
    )


def build_loss_fn(cfg: TrainAppConfig):
    return get_loss(
        cfg.train.loss.name,
        **cfg.train.loss.params,
    )


def build_train_callbacks(
    cfg: TrainAppConfig,
    *,
    main_process: bool,
    wandb_name: str | None = None,
    raw_config=None,
):
    callbacks = [ProgressCallback()]

    if not main_process:
        return callbacks

    cfg.checkpoint_root.mkdir(parents=True, exist_ok=True)
    callbacks.insert(
        0,
        CheckpointCallback(
            save_dir=str(cfg.checkpoint_root),
            every=cfg.train.checkpoint.every,
            save_best=cfg.train.checkpoint.save_best,
            monitor=cfg.train.checkpoint.monitor,
            mode=cfg.train.checkpoint.mode,
        ),
    )
    callbacks.insert(1, LoggingCallback())

    if cfg.wandb.mode != "disabled":
        callbacks.append(
            WandbCallback(
                project=cfg.wandb.project,
                name=wandb_name,
                config=OmegaConf.to_container(raw_config, resolve=True) if raw_config is not None else None,
                mode=cfg.wandb.mode,
            )
        )

    return callbacks


def build_eval_callbacks(cfg: EvaluateAppConfig | InferAppConfig):
    callbacks = []

    if cfg.mode == "evaluation":
        callbacks.append(MetricsCallback(verbose=True))

    if cfg.save_results:
        callbacks.append(
            SaveResultsCallback(
                save_dir=str(cfg.results_root),
                mean=cfg.dataset.normalize.mean,
                std=cfg.dataset.normalize.std,
            )
        )

    return callbacks


def load_eval_mask(mask_path: str | None, logger=None):
    if not mask_path:
        return None

    path = Path(mask_path)
    if not path.exists():
        if logger is not None:
            logger.warning(f"掩码文件不存在: {mask_path}")
        return None

    mask = np.load(path)
    if logger is not None:
        logger.info(f"加载评估掩码: {mask_path}")
    return mask
