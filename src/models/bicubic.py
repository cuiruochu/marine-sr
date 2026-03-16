"""
Bicubic 基线模型

无实际模型，仅用于评估对比。直接使用双三次插值。
"""
from .registry import register_model
from .base import create_model_result


@register_model("Bicubic", default_params={})
def create_bicubic(params: dict, in_dim: int, upscale: int):
    """
    Bicubic 模型工厂。

    Bicubic 无需神经网络模型，返回 None。
    测试时直接使用双三次插值。

    Args:
        params: 参数字典（未使用）
        in_dim: 输入通道数（未使用）
        upscale: 放大倍数（未使用）

    Returns:
        {"model": None, "model_name": "Bicubic"}
    """
    return create_model_result(None, "Bicubic")