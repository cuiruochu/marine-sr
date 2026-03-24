"""模型装配器。"""

from src.models.factory import create_model


def build_model_bundle(cfg):
    return create_model(cfg.models_build_spec)
