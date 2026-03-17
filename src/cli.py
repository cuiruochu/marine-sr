"""
统一训练入口

自动检测单卡/分布式模式，无需两个脚本。

用法:
    # 单卡训练
    uv run python src/cli.py

    # 分布式训练（使用 torchrun）
    torchrun --nproc_per_node=4 src/cli.py

    # 切换模型和数据集
    uv run python src/cli.py model=rcan dataset=mwd

    # 命令行覆盖
    uv run python src/cli.py train.epochs=100 train.lr=1e-4

    # 分布式训练带参数
    torchrun --nproc_per_node=4 src/cli.py model=edsr train.epochs=200
"""

import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from src.core import Engine
from src.models import create_model
from src.callbacks import (
    CheckpointCallback,
    WandbCallback,
    LoggingCallback,
    ProgressCallback,
)
from src.losses import get_loss
from src.optim import get_optimizer, get_scheduler
from src.utils import init_logger, get_logger, PROJECT_ROOT
from src.utils.distributed import (
    is_distributed,
    init_distributed,
    cleanup_distributed,
    is_main_process,
    get_rank,
    get_world_size,
)
from src.data import set_epoch_for_sampler


class ConfigAdapter:
    """
    配置适配器

    将 Hydra DictConfig 转换为兼容现有接口的对象。
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        # 模型配置
        self.model_name = cfg.model.name
        self.model_params = dict(cfg.model.params)

        # 数据集配置
        self.marine_param = cfg.dataset.param
        self.upscale = cfg.dataset.upscale
        self.lr_patch_size = cfg.dataset.lr_patch_size
        self.in_dim = cfg.dataset.channels
        self.mean = list(cfg.dataset.normalize.mean)
        self.std = list(cfg.dataset.normalize.std)

        # 训练配置
        self.epochs = cfg.train.epochs
        self.batch_size = cfg.train.batch_size
        self.lr = cfg.train.lr

        # 优化器配置
        self.optimizer_name = cfg.train.optimizer.name
        self.optimizer_params = dict(cfg.train.optimizer.params)

        # 调度器配置
        self.scheduler_name = cfg.train.scheduler.name
        self.scheduler_params = dict(cfg.train.scheduler.params)

        # 损失函数配置
        self.loss_name = cfg.train.loss.name
        self.loss_params = dict(getattr(cfg.train.loss, "params", {}))

        # 路径配置
        self.checkpoint_dir = cfg.paths.checkpoint_dir


def get_data_loaders(cfg: DictConfig, adapter: ConfigAdapter):
    """
    创建数据加载器

    自动处理分布式采样器。

    Args:
        cfg: Hydra 配置
        adapter: 配置适配器

    Returns:
        (train_loader, val_loader)
    """
    from src.datasets import get_loader
    from dataclasses import dataclass, field
    from typing import List

    @dataclass
    class TrainConfig:
        marine_param: str = ""
        upscale: int = 2
        lr_patch_size: int = 30
        batch_size: int = 16
        mean: List[float] = field(default_factory=list)
        std: List[float] = field(default_factory=list)
        train_hr_root: str = ""
        val_hr_root: str = ""

    # 构建数据路径
    data_root = PROJECT_ROOT / "data" / adapter.marine_param

    train_cfg = TrainConfig(
        marine_param=adapter.marine_param,
        upscale=adapter.upscale,
        lr_patch_size=adapter.lr_patch_size,
        batch_size=adapter.batch_size,
        mean=adapter.mean,
        std=adapter.std,
        train_hr_root=str(data_root / "Train" / adapter.marine_param),
        val_hr_root=str(data_root / "Val" / adapter.marine_param),
    )

    return get_loader(train_cfg, batch_size=adapter.batch_size)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # 检测并初始化分布式环境
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_ddp = world_size > 1

    if is_ddp:
        init_distributed()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 设置随机种子（分布式时每个进程不同）
    seed = cfg.seed + (get_rank() if is_ddp else 0)
    torch.manual_seed(seed)

    # 创建配置适配器
    adapter = ConfigAdapter(cfg)

    # 只在主进程打印日志
    if is_main_process():
        init_logger()
        logger = get_logger()

        mode_str = f"分布式训练 ({get_world_size()} GPUs)" if is_ddp else "单卡训练"
        logger.info(f"{mode_str}")
        logger.info("配置信息:")
        logger.info(f"  模型: {adapter.model_name}")
        logger.info(f"  参数: {adapter.marine_param}")
        logger.info(f"  放大倍数: {adapter.upscale}")
        logger.info(f"  训练轮数: {adapter.epochs}")
        logger.info(f"  批次大小: {adapter.batch_size}")
        logger.info(f"  学习率: {adapter.lr}")
        logger.info(f"  优化器: {adapter.optimizer_name}")
        logger.info(f"  调度器: {adapter.scheduler_name}")
        logger.info(f"  损失函数: {adapter.loss_name}")
    else:
        logger = None

    # 创建模型
    if is_main_process():
        logger.info("创建模型...")

    model_dict = create_model(adapter)
    model = model_dict["model"]

    if is_main_process():
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  可训练参数: {total_params:,}")

    # 创建数据加载器
    if is_main_process():
        logger.info("创建数据加载器...")

    train_loader, val_loader = get_data_loaders(cfg, adapter)

    # 创建组件
    optimizer = get_optimizer(
        adapter.optimizer_name,
        model,
        lr=adapter.lr,
        **adapter.optimizer_params
    )

    scheduler = get_scheduler(
        adapter.scheduler_name,
        optimizer,
        **adapter.scheduler_params
    )

    loss_fn = get_loss(adapter.loss_name, **adapter.loss_params)

    # 创建回调（只在主进程保存检查点）
    callbacks = []

    if is_main_process():
        checkpoint_dir = (
            PROJECT_ROOT / cfg.paths.checkpoint_dir
            / adapter.model_name
            / adapter.marine_param
            / f"x{adapter.upscale}"
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        callbacks.extend([
            CheckpointCallback(
                save_dir=str(checkpoint_dir),
                every=cfg.train.checkpoint.every,
                save_best=cfg.train.checkpoint.save_best,
                monitor=cfg.train.checkpoint.monitor,
                mode=cfg.train.checkpoint.mode,
            ),
            LoggingCallback(),
        ])

    callbacks.append(ProgressCallback())

    # WandB
    if cfg.wandb.mode != "disabled" and is_main_process():
        import wandb

        output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        exp_name = output_dir.name

        wandb.init(
            project=cfg.wandb.project,
            mode=cfg.wandb.mode,
            name=exp_name,
            config=OmegaConf.to_container(cfg),
        )
        callbacks.append(WandbCallback(mode=cfg.wandb.mode))

    # 创建引擎
    engine = Engine(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        device=device,
        callbacks=callbacks,
        ddp=is_ddp,
    )

    # 开始训练
    if is_main_process():
        logger.info("开始训练...")

    # 分布式训练需要设置 sampler epoch
    if is_ddp:
        engine.callbacks.on_train_begin(engine)
        history = {"train_loss": [], "val_psnr": [], "val_ssim": [], "val_mae": []}

        for epoch in range(1, adapter.epochs + 1):
            engine.current_epoch = epoch
            set_epoch_for_sampler(train_loader, epoch)

            engine.callbacks.on_epoch_begin(engine, epoch)
            engine._train_loader = train_loader
            epoch_loss = engine._train_epoch(train_loader)

            val_logs = engine.evaluate(val_loader, adapter.mean, adapter.std)
            val_logs["lr"] = engine.lr
            val_logs["train_loss"] = epoch_loss

            if engine.scheduler is not None:
                engine.scheduler.step()

            engine.callbacks.on_epoch_end(engine, epoch, val_logs)

            history["train_loss"].append(epoch_loss)
            history["val_psnr"].append(val_logs["val_psnr"])
            history["val_ssim"].append(val_logs["val_ssim"])
            history["val_mae"].append(val_logs["val_mae"])

        engine.callbacks.on_train_end(engine)
    else:
        # 单卡训练
        engine.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=adapter.epochs,
            mean=adapter.mean,
            std=adapter.std,
        )

    # 清理
    if is_ddp:
        cleanup_distributed()

    if is_main_process():
        if cfg.wandb.mode != "disabled":
            import wandb
            wandb.finish()
        logger.info("训练完成")


if __name__ == "__main__":
    main()