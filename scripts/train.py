"""
训练入口脚本

使用 YAML 配置文件驱动训练，支持实验管理和日志记录。
"""

import os
import wandb

from configs import get_train_config, setup_seed
from src.datasets import get_loader
from src.models import create_model
from src.trainers import get_trainer
from src.utils import create_experiment, init_logger, get_logger


def main():
    # 配置
    train_cfg = get_train_config("configs/train.yaml")

    # 初始化实验管理
    experiment = create_experiment(train_cfg, base_dir="experiments")

    # 初始化日志
    log_file = experiment.exp_dir / "train.log"
    init_logger(log_file=str(log_file))
    logger = get_logger()

    # 固定随机种子
    setup_seed(0)

    logger.info(f"实验目录: {experiment.exp_dir}")
    logger.info(f"模型: {train_cfg.model_name}")
    logger.info(f"参数: {train_cfg.marine_param}")
    logger.info(f"放大倍数: {train_cfg.upscale}")

    # 初始化 wandb
    os.environ["WANDB_MODE"] = "offline"
    wandb_log = wandb.init(
        project="Marine-Parameters-SupervisedSR",
        name=experiment.get_wandb_name(),  # 与实验目录名一致
        config=train_cfg.raw
    )

    # 数据加载器
    logger.info("创建数据加载器...")
    train_loader, val_loader = get_loader(train_cfg, batch_size=train_cfg.batch_size)

    # 模型
    logger.info("创建模型...")
    model = create_model(train_cfg)
    logger.info(f"模型名: {model['model_name']}")

    # 训练器
    trainer = get_trainer(
        train_cfg=train_cfg,
        model_dict=model,
        train_loader=train_loader,
        val_loader=val_loader,
        wandb_log=wandb_log
    )

    # 加载预训练权重
    if train_cfg.checkpoint:
        logger.info(f"加载预训练权重: {train_cfg.checkpoint}")
        trainer.load_weight(train_cfg.checkpoint)

    # 打印模型信息
    trainer.print_model_info()

    # 开始训练
    logger.info("开始训练...")
    trainer.fit()

    wandb.finish()
    logger.info("训练完成")


if __name__ == "__main__":
    main()