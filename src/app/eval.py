"""评估编排入口。"""

import logging

import numpy as np
from omegaconf import DictConfig

from src.app.checkpoint_validation import validate_checkpoint_matches_config
from src.app.config_parsing import load_eval_task_config
from src.app.config_types import EvalTaskConfig, EvaluateAppConfig, InferAppConfig
from src.app.eval_builders import build_eval_callbacks, build_test_loader
from src.app.model_builders import build_model_bundle
from src.app.runtime import init_task_logger
from src.core.evaluator import Evaluator
from src.datamodules.validation import validate_eval_runtime_inputs
from src.utils.path import resolve_project_path

logger = logging.getLogger(__name__)


def run_eval_task(raw_cfg: DictConfig | EvalTaskConfig):
    cfg = raw_cfg if isinstance(raw_cfg, (EvaluateAppConfig, InferAppConfig)) else load_eval_task_config(raw_cfg)

    init_task_logger("evaluate.log" if cfg.mode == "evaluation" else "infer.log")

    checkpoint = cfg.checkpoint
    if not checkpoint:
        raise ValueError("必须指定检查点路径: checkpoint=./path/to/checkpoint.pth")

    validate_eval_runtime_inputs(cfg)

    checkpoint_path = resolve_project_path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
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

    normalize = cfg.test_loader_spec.normalize
    mask = _load_mask(cfg.mask)
    logger.info("开始配对评估..." if cfg.mode == "evaluation" else "开始推理...")
    evaluator.run(
        test_loader=test_loader,
        mean=normalize.mean,
        std=normalize.std,
        mask=mask,
        mode=cfg.mode,
    )
    logger.info("配对评估完成" if cfg.mode == "evaluation" else "推理完成")


def run_evaluation(raw_cfg: DictConfig | EvaluateAppConfig):
    return run_eval_task(raw_cfg)


def run_inference(raw_cfg: DictConfig | InferAppConfig):
    return run_eval_task(raw_cfg)


def _log_evaluation_configuration(logger, cfg: EvaluateAppConfig | InferAppConfig):
    logger.info("评估配置:")
    logger.info(f"  模型: {cfg.models.name}")
    logger.info(f"  数据集: {cfg.dataset.name}")
    logger.info(f"  放大倍数: {cfg.dataset.upscale}")
    logger.info(f"  模式: {cfg.mode}")
    logger.info(f"  检查点: {cfg.checkpoint}")
    logger.info(f"  Mask: {cfg.mask}" if cfg.mask else "  Mask: None")
    logger.info("创建模型...")


def _load_mask(mask_path: str | None) -> np.ndarray | None:
    if not mask_path:
        return None
    mask = np.load(resolve_project_path(mask_path))
    if mask.dtype != np.bool_:
        raise TypeError(f"mask 文件必须是 bool 类型，当前 dtype={mask.dtype}")
    if mask.ndim != 2:
        raise ValueError(f"mask 文件 shape 必须是 (H, W)，当前={mask.shape}")
    return mask
