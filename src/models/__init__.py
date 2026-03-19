"""
模型创建模块 - 使用注册机制

提供基于装饰器的模型注册，添加新模型只需在模型文件中加 @register_model 装饰器。
"""

# 先导入注册机制（必须在模型导入之前）
from .registry import (
    register_model,
    get_model_info,
    list_models,
    MODEL_REGISTRY
)
from .base import create_model_result, merge_params

# 导入基础模型（触发装饰器注册）
from .EDSR import EDSR
from .RCAN import RCAN
from .RDN import RDN
from .SwinIR import SwinIR
from .MySR import MySR
from .MySRAb import MySRAb1, MySRAb2
from .bicubic import create_bicubic

OPTIONAL_MODEL_IMPORT_ERRORS = {}

try:
    from .ATD import ATD
except ImportError as exc:
    OPTIONAL_MODEL_IMPORT_ERRORS["ATD"] = exc

try:
    from .CAMixer import CAMixerSR
except ImportError as exc:
    OPTIONAL_MODEL_IMPORT_ERRORS["CAMixer"] = exc


def create_model(config):
    """
    基于配置创建模型实例。

    Args:
        config: TrainConfig 或 TestConfig，包含以下字段：
            - model_name: 模型名称（如 "EDSR"）
            - model_params: 模型参数字典（可选）
            - in_dim: 输入通道数
            - upscale: 放大倍数

    Returns:
        dict: {"model": model, "model_name": str}
    """
    model_name = config.model_name
    model_params = getattr(config, 'model_params', {}) or {}

    if model_name in OPTIONAL_MODEL_IMPORT_ERRORS and model_name not in MODEL_REGISTRY:
        raise ModuleNotFoundError(
            f"模型 '{model_name}' 依赖未安装，请使用 `uv sync --extra all` 安装可选依赖。"
        ) from OPTIONAL_MODEL_IMPORT_ERRORS[model_name]

    # 从注册表获取模型信息
    info = get_model_info(model_name)

    if info.factory is None:
        raise ValueError(f"模型 '{model_name}' 不支持创建。")

    params = merge_params(info.default_params or {}, model_params)
    return info.factory(params, config.in_dim, config.upscale)


# 导出公共 API
__all__ = [
    # 注册机制
    "register_model",
    "list_models",
    "MODEL_REGISTRY",

    # 创建函数
    "create_model",
    "create_model_result",
    "merge_params",
    "OPTIONAL_MODEL_IMPORT_ERRORS",

    # 模型类（供直接导入）
    "EDSR", "RCAN", "RDN", "SwinIR",
    "MySR", "MySRAb1", "MySRAb2",
]

if "ATD" in MODEL_REGISTRY:
    __all__.append("ATD")

if "CAMixer" in MODEL_REGISTRY:
    __all__.append("CAMixerSR")
