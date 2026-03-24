"""模型工厂。"""

from .base import merge_params
from .registry import OPTIONAL_MODEL_IMPORT_ERRORS, get_model_info, normalize_model_name


def create_model(config):
    model_name = normalize_model_name(config.model_name)
    model_params = getattr(config, "model_params", {}) or {}
    upscale = int(getattr(config, "upscale"))
    in_dim = int(getattr(config, "in_dim"))

    if model_name in OPTIONAL_MODEL_IMPORT_ERRORS:
        raise ModuleNotFoundError(
            f"模型 '{model_name}' 依赖未安装，请使用 `uv sync --extra all` 安装可选依赖。"
        ) from OPTIONAL_MODEL_IMPORT_ERRORS[model_name]

    info = get_model_info(model_name)
    if info.factory is None:
        raise ValueError(f"模型 '{model_name}' 不支持创建。")

    params = merge_params(info.default_params or {}, model_params)
    return info.factory(params, in_dim, upscale)
