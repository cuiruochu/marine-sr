"""
训练编排入口
"""

from dataclasses import asdict
import os
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.app.builders import (
    build_loss_fn,
    build_model_bundle,
    build_optimizer,
    build_scheduler,
    build_train_callbacks,
    build_train_loaders,
)
from src.app.checkpoint_validation import validate_checkpoint_matches_config
from src.app.config import TrainAppConfig, load_train_config
from src.core import Engine
from src.data import set_epoch_for_sampler, validate_train_runtime_inputs
from src.utils import PROJECT_ROOT, get_logger, init_logger, seed_everything
from src.utils.distributed import (
    cleanup_distributed,
    get_rank,
    get_world_size,
    init_distributed,
    is_main_process,
)


def run_training(raw_cfg: DictConfig | TrainAppConfig):
    cfg = raw_cfg if isinstance(raw_cfg, TrainAppConfig) else load_train_config(raw_cfg)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_ddp = world_size > 1

    if is_ddp:
        init_distributed()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    seed = cfg.seed + (get_rank() if is_ddp else 0)
    seed_everything(seed)

    logger = _init_train_logger(is_ddp)
    if logger is not None:
        _log_training_configuration(logger, cfg)

    validate_train_runtime_inputs(cfg)

    model_bundle = build_model_bundle(cfg)
    model = model_bundle["model"]

    if logger is not None:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  可训练参数: {total_params:,}")
        logger.info("创建数据加载器...")

    train_loader, val_loader = build_train_loaders(cfg)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    loss_fn = build_loss_fn(cfg)

    wandb_name = None
    if cfg.wandb.mode != "disabled" and is_main_process():
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        wandb_name = os.path.basename(output_dir)

    callbacks = build_train_callbacks(
        cfg,
        main_process=is_main_process(),
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
        ddp=is_ddp,
        config_snapshot=_build_config_snapshot(raw_cfg, cfg),
    )

    start_epoch = 1
    remaining_epochs = cfg.train.epochs
    if cfg.resume.checkpoint:
        checkpoint_path = _resolve_project_path(cfg.resume.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"恢复训练检查点不存在: {checkpoint_path}")
        validate_checkpoint_matches_config(cfg, checkpoint_path)

        loaded_epoch = engine.load_checkpoint(
            str(checkpoint_path),
            load_optimizer=cfg.resume.load_optimizer,
            load_scheduler=cfg.resume.load_scheduler,
            load_rng_state=cfg.resume.load_rng_state,
        )
        start_epoch = loaded_epoch + 1
        remaining_epochs = cfg.train.epochs - loaded_epoch

        if logger is not None:
            logger.info(f"从检查点恢复训练: {checkpoint_path}")
            logger.info(f"  已完成轮数: {loaded_epoch}")
            logger.info(f"  剩余轮数: {max(remaining_epochs, 0)}")

        if remaining_epochs <= 0:
            if logger is not None:
                logger.info("配置中的总训练轮数不大于检查点轮数，无需继续训练")
            if is_ddp:
                cleanup_distributed()
            return

    if logger is not None:
        logger.info("开始训练...")

    mean = cfg.dataset.normalize.mean
    std = cfg.dataset.normalize.std

    if is_ddp:
        _fit_distributed(engine, train_loader, val_loader, remaining_epochs, mean, std, start_epoch=start_epoch)
    else:
        engine.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=remaining_epochs,
            mean=mean,
            std=std,
            start_epoch=start_epoch,
        )

    if is_ddp:
        cleanup_distributed()

    if logger is not None:
        logger.info("训练完成")


def _init_train_logger(is_ddp: bool):
    if not is_main_process():
        return None

    init_logger()
    logger = get_logger()
    mode_str = f"分布式训练 ({get_world_size()} GPUs)" if is_ddp else "单卡训练"
    logger.info(mode_str)
    logger.info("配置信息:")
    return logger


def _log_training_configuration(logger, cfg: TrainAppConfig):
    logger.info(f"  模型: {cfg.model.name}")
    logger.info(f"  数据集: {cfg.dataset.name}")
    logger.info(f"  放大倍数: {cfg.dataset.upscale}")
    logger.info(f"  训练轮数: {cfg.train.epochs}")
    logger.info(f"  批次大小: {cfg.train.batch_size}")
    logger.info(f"  学习率: {cfg.train.lr}")
    logger.info(f"  优化器: {cfg.train.optimizer.name}")
    logger.info(f"  调度器: {cfg.train.scheduler.name}")
    logger.info(f"  损失函数: {cfg.train.loss.name}")
    logger.info("创建模型...")


def _fit_distributed(engine: Engine, train_loader, val_loader, epochs: int, mean, std, start_epoch: int = 1):
    engine.callbacks.on_train_begin(engine)

    for epoch in range(start_epoch, start_epoch + epochs):
        engine.current_epoch = epoch
        set_epoch_for_sampler(train_loader, epoch)

        engine.callbacks.on_epoch_begin(engine, epoch)
        engine._train_loader = train_loader
        epoch_loss = engine._train_epoch(train_loader)

        val_logs = engine.evaluate(val_loader, mean, std)
        val_logs["lr"] = engine.lr
        val_logs["train_loss"] = epoch_loss

        if engine.scheduler is not None:
            engine.scheduler.step()

        engine.callbacks.on_epoch_end(engine, epoch, val_logs)

    engine.callbacks.on_train_end(engine)


def _build_config_snapshot(raw_cfg: DictConfig | TrainAppConfig, cfg: TrainAppConfig) -> dict:
    if isinstance(raw_cfg, DictConfig):
        return OmegaConf.to_container(raw_cfg, resolve=True)
    return asdict(cfg)


def _resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
