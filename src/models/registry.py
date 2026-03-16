"""
模型注册机制

提供装饰器方式的模型注册，避免 if-elif 链。
添加新模型只需在模型文件中加装饰器，无需修改 __init__.py。
"""

from typing import Callable, Dict, Any, Optional
from dataclasses import dataclass


# 工厂函数类型别名
ModelFactory = Callable[[dict, int, int], Dict[str, Any]]
# 参数: (model_params, in_dim, upscale)
# 返回: {"model": model, "model_name": str}


@dataclass
class ModelInfo:
    """注册模型的信息"""
    name: str
    factory: Optional[ModelFactory] = None
    default_params: dict = None


# 全局注册表
MODEL_REGISTRY: Dict[str, ModelInfo] = {}


def register_model(name: str, default_params: dict = None):
    """
    注册单参数模型的装饰器。

    Args:
        name: 模型名称（如 "EDSR", "RCAN"）
        default_params: 默认参数字典

    Usage:
        @register_model("EDSR", default_params={"n_feats": 64, "n_resblocks": 16})
        def create_edsr(params, in_dim, upscale):
            ...

    Returns:
        装饰器函数
    """
    def decorator(factory: ModelFactory):
        MODEL_REGISTRY[name] = ModelInfo(
            name=name,
            factory=factory,
            default_params=default_params or {}
        )
        return factory
    return decorator


def get_model_info(name: str) -> ModelInfo:
    """
    从注册表获取模型信息。

    Args:
        name: 模型名称

    Returns:
        ModelInfo 实例

    Raises:
        ValueError: 模型未注册
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"模型 '{name}' 未注册。可用模型: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name]


def list_models() -> list:
    """列出所有已注册模型"""
    return list(MODEL_REGISTRY.keys())