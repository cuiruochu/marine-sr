"""
评估编排入口
"""

from pathlib import Path

from omegaconf import DictConfig

from src.app.builders import (
    build_eval_callbacks,
    build_model_bundle,
    build_test_loader,
    load_eval_mask,
)
from src.app.checkpoint_validation import validate_checkpoint_matches_config
from src.app.config import EvaluateAppConfig, InferAppConfig, load_eval_task_config
from src.core import Evaluator
from src.data import validate_eval_runtime_inputs
from src.utils import get_logger, init_logger


def run_evaluation(raw_cfg: DictConfig | EvaluateAppConfig | InferAppConfig):
    cfg = raw_cfg if isinstance(raw_cfg, (EvaluateAppConfig, InferAppConfig)) else load_eval_task_config(raw_cfg)

    init_logger()
    logger = get_logger()

    checkpoint = cfg.checkpoint
    if not checkpoint:
        logger.error("必须指定检查点路径: checkpoint=./path/to/checkpoint.pth")
        return

    validate_eval_runtime_inputs(cfg)

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"检查点文件不存在: {checkpoint}")
        return
    validate_checkpoint_matches_config(cfg, checkpoint_path)

    _log_evaluation_configuration(logger, cfg)

    model_bundle = build_model_bundle(cfg)
    model = model_bundle["model"]
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  可训练参数: {total_params:,}")

    callbacks = build_eval_callbacks(cfg)
    if cfg.save_results:
        logger.info(f"  结果保存目录: {cfg.results_root}")

    evaluator = Evaluator(model=model, callbacks=callbacks)
    logger.info(f"加载检查点: {checkpoint}")
    evaluator.load_checkpoint(str(checkpoint_path))

    logger.info("创建推理/评估数据加载器...")
    test_loader = build_test_loader(cfg)
    logger.info(f"  样本数: {len(test_loader)}")

    mask = load_eval_mask(cfg.eval_mask, logger) if cfg.mode == "evaluation" else None
    logger.info("开始配对评估..." if cfg.mode == "evaluation" else "开始推理...")
    evaluator.run(
        test_loader=test_loader,
        mean=cfg.dataset.normalize.mean,
        std=cfg.dataset.normalize.std,
        mask=mask,
    )
    logger.info("配对评估完成" if cfg.mode == "evaluation" else "推理完成")


def _log_evaluation_configuration(logger, cfg: EvaluateAppConfig | InferAppConfig):
    logger.info("评估配置:")
    logger.info(f"  模型: {cfg.model.name}")
    logger.info(f"  数据集: {cfg.dataset.name}")
    logger.info(f"  放大倍数: {cfg.dataset.upscale}")
    logger.info(f"  模式: {cfg.mode}")
    logger.info(f"  检查点: {cfg.checkpoint}")
    logger.info("创建模型...")
