"""训练链路装配器。"""

from omegaconf import OmegaConf

from src.app.config_types import TrainAppConfig
from src.callbacks.checkpoint import CheckpointCallback
from src.callbacks.logging import LoggingCallback
from src.callbacks.wandb import WandbCallback
from src.datamodules.train_builders import build_train_val_loaders
from src.losses.registry import get_loss
from src.optim.optimizer import get_optimizer
from src.optim.scheduler import get_scheduler


def build_train_loaders(cfg: TrainAppConfig):
    return build_train_val_loaders(
        cfg.train_loader_spec,
        train_batch_size=cfg.train.batch_size,
        train_num_workers=cfg.train.num_workers,
        eval_num_workers=cfg.train.val_num_workers,
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
    return get_loss(cfg.train.loss.name, **cfg.train.loss.params)


def build_train_callbacks(
    cfg: TrainAppConfig,
    *,
    main_process: bool,
    wandb_name: str | None = None,
    raw_config=None,
):
    if not main_process:
        return []

    cfg.checkpoint_root.mkdir(parents=True, exist_ok=True)
    callbacks = [
        CheckpointCallback(
            save_dir=str(cfg.checkpoint_root),
            every=cfg.train.checkpoint.every,
            save_best=cfg.train.checkpoint.save_best,
            monitor=cfg.train.checkpoint.monitor,
            mode=cfg.train.checkpoint.mode,
        ),
        LoggingCallback(),
    ]

    if cfg.wandb.mode == "disabled":
        return callbacks

    callbacks.append(
        WandbCallback(
            project=cfg.wandb.project,
            name=wandb_name,
            config=OmegaConf.to_container(raw_config, resolve=True) if raw_config is not None else None,
            mode=cfg.wandb.mode,
        )
    )
    return callbacks
