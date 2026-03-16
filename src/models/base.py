"""
模型创建工具函数
"""

from typing import Dict, Any


def create_model_result(
    model,
    model_name: str,
    mwd_encoder=None,
    other_encoder=None,
    mwd_decoder=None,
    other_decoder=None
) -> Dict[str, Any]:
    """
    创建标准化的模型返回字典。

    对于单模型: 返回 {"model": model, "model_name": model_name}
    对于统一模型: 额外包含 encoder/decoder 组件

    Args:
        model: 模型实例
        model_name: 模型名称字符串（用于 checkpoint 命名）
        mwd_encoder: MWD 参数编码器（统一模型用）
        other_encoder: 其他参数编码器（统一模型用）
        mwd_decoder: MWD 参数解码器（统一模型用）
        other_decoder: 其他参数解码器（统一模型用）

    Returns:
        标准化的模型字典
    """
    result = {"model": model, "model_name": model_name}

    if mwd_encoder is not None:
        result.update({
            "mwd_encoder": mwd_encoder,
            "other_encoder": other_encoder,
            "mwd_decoder": mwd_decoder,
            "other_decoder": other_decoder
        })

    return result


def merge_params(default_params: dict, user_params: dict) -> dict:
    """
    合并默认参数和用户参数。

    用户参数优先级高于默认参数。

    Args:
        default_params: 默认参数字典
        user_params: 用户提供的参数字典（可为 None）

    Returns:
        合并后的参数字典
    """
    result = default_params.copy()
    result.update(user_params or {})
    return result