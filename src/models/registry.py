"""模型注册机制。"""

from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Callable, Dict, Optional

ModelFactory = Callable[[dict, int, int], Dict[str, Any]]


@dataclass
class ModelInfo:
    name: str
    factory: Optional[ModelFactory] = None
    default_params: Dict[str, Any] = field(default_factory=dict)


MODEL_REGISTRY: Dict[str, ModelInfo] = {}
OPTIONAL_MODEL_IMPORT_ERRORS: Dict[str, ImportError] = {}
_BUILTIN_MODELS_REGISTERED = False


def register_model(name: str, default_params: Optional[Dict[str, Any]] = None):
    """注册模型的装饰器。"""

    def decorator(factory: ModelFactory):
        MODEL_REGISTRY[name] = ModelInfo(name=name, factory=factory, default_params=default_params or {})
        return factory

    return decorator


def _import_optional_module(module_name: str, model_name: str) -> None:
    try:
        import_module(f"{__package__}.{module_name}")
    except ImportError as exc:
        OPTIONAL_MODEL_IMPORT_ERRORS[model_name] = exc


def ensure_builtin_models_registered() -> None:
    global _BUILTIN_MODELS_REGISTERED
    if _BUILTIN_MODELS_REGISTERED:
        return

    for module_name in ("bicubic", "edsr", "mysr", "mysrab", "rcan", "rdn"):
        import_module(f"{__package__}.{module_name}")

    _import_optional_module("swinir", "SwinIR")
    _import_optional_module("atd", "ATD")
    _import_optional_module("camixer", "CAMixer")
    _BUILTIN_MODELS_REGISTERED = True


def get_model_info(name: str) -> ModelInfo:
    ensure_builtin_models_registered()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"模型 '{name}' 未注册。可用模型: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]


def list_models() -> list[str]:
    ensure_builtin_models_registered()
    return list(MODEL_REGISTRY.keys())
