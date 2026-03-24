"""评估/推理数据集与加载器装配器。"""

from src.app.config_types import EvaluationLoaderSpec, InferenceLoaderSpec, TestLoaderSpec
from src.datasets.marine import MarineEvalDataset, MarineInferDataset


def build_evaluation_dataset(spec: EvaluationLoaderSpec):
    """构建配对评估数据集。"""
    pair = spec.eval_pair
    return MarineEvalDataset(
        lr_root=pair.lr_root,
        hr_root=pair.hr_root,
        upscale=spec.upscale,
        mean=pair.normalize.mean,
        std=pair.normalize.std,
        sample_limit=pair.max_sample,
        return_filename=True,
    )


def build_inference_dataset(spec: InferenceLoaderSpec):
    """构建离线推理数据集。"""
    return MarineInferDataset(
        lr_root=spec.infer_lr_root,
        upscale=spec.upscale,
        mean=spec.normalize.mean,
        std=spec.normalize.std,
        sample_limit=False,
        return_filename=True,
    )


def build_test_dataset(spec: TestLoaderSpec):
    """构建评估或推理数据集。"""
    if isinstance(spec, EvaluationLoaderSpec):
        return build_evaluation_dataset(spec)
    return build_inference_dataset(spec)


def build_test_loader(spec: TestLoaderSpec, batch_size: int, num_workers: int):
    """构建评估或推理数据加载器。"""
    from .loaders import build_eval_dataloader

    test_set = build_test_dataset(spec)
    return build_eval_dataloader(test_set, batch_size=batch_size, num_workers=num_workers)
