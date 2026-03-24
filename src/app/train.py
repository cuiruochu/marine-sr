"""训练编排入口。"""

import logging
import os

import torch
from omegaconf import DictConfig

from src.app.checkpoint_validation import validate_checkpoint_matches_config
from src.app.config_parsing import load_train_config
from src.app.config_types import TrainAppConfig
from src.app.model_builders import build_model_bundle
from src.app.runtime import build_config_snapshot, hydra_run_name, init_task_logger
from src.app.train_builders import (
    build_loss_fn,
    build_optimizer,
    build_scheduler,
    build_train_callbacks,
    build_train_loaders,
)
from src.core.engine import Engine
from src.datamodules.validation import validate_train_runtime_inputs
from src.utils.path import resolve_project_path
from src.utils.random_state import seed_everything

logger = logging.getLogger(__name__)


def run_training(raw_cfg: DictConfig | TrainAppConfig):
    cfg = raw_cfg if isinstance(raw_cfg, TrainAppConfig) else load_train_config(raw_cfg)

    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        raise NotImplementedError("当前版本仅支持单进程单卡训练，不支持多进程启动")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(cfg.seed)

    init_task_logger("train.log")
    logger.info("单卡训练")
    logger.info("配置信息:")
    _log_training_configuration(logger, cfg)

    validate_train_runtime_inputs(cfg)

    model_bundle = build_model_bundle(cfg)
    model = model_bundle["model"]

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  可训练参数: {total_params:,}")
    logger.info("创建数据加载器...")

    train_loader, val_loader = build_train_loaders(cfg)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    loss_fn = build_loss_fn(cfg)

    wandb_name = hydra_run_name() if cfg.wandb.mode != "disabled" else None

    callbacks = build_train_callbacks(
        cfg,
        main_process=True,
        wandb_name=wandb_name,
        raw_config=raw_cfg if not isinstance(raw_cfg, TrainAppConfig) else None,
    )

    engine = Engine(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        device=device,
        callbacks=callbacks,
        config_snapshot=build_config_snapshot(raw_cfg, cfg),
    )

    start_epoch = 1
    remaining_epochs = cfg.train.epochs
    if cfg.resume.checkpoint:
        checkpoint_path = resolve_project_path(cfg.resume.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"恢复训练检查点不存在: {checkpoint_path}")
        validate_checkpoint_matches_config(cfg, checkpoint_path)

        loaded_epoch = engine.load_checkpoint(
            str(checkpoint_path),
            load_optimizer=cfg.resume.load_optimizer,
            load_scheduler=cfg.resume.load_scheduler,
            load_callbacks=cfg.resume.load_callbacks,
            load_rng_state=cfg.resume.load_rng_state,
        )
        start_epoch = loaded_epoch + 1
        remaining_epochs = cfg.train.epochs - loaded_epoch

        logger.info(f"从检查点恢复训练: {checkpoint_path}")
        logger.info(f"  已完成轮数: {loaded_epoch}")
        logger.info(f"  剩余轮数: {max(remaining_epochs, 0)}")

        if remaining_epochs <= 0:
            logger.info("配置中的总训练轮数不大于检查点轮数，无需继续训练")
            return

    logger.info("开始训练...")

    engine.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=remaining_epochs,
        start_epoch=start_epoch,
    )

    logger.info("训练完成")


def _log_training_configuration(logger, cfg: TrainAppConfig):
    logger.info(f"  模型: {cfg.models.name}")
    logger.info(f"  数据集: {cfg.dataset.name}")
    logger.info(f"  放大倍数: {cfg.upscale}")
    logger.info(f"  训练轮数: {cfg.train.epochs}")
    logger.info(f"  批次大小: {cfg.train.batch_size}")
    logger.info(f"  学习率: {cfg.train.lr}")
    logger.info(f"  优化器: {cfg.train.optimizer.name}")
    logger.info(f"  调度器: {cfg.train.scheduler.name}")
    logger.info(f"  损失函数: {cfg.train.loss.name}")
    logger.info("创建模型...")
