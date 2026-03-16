"""
模型单元测试

测试所有模型的前向传播、输出 shape、输入通道等。
"""

import pytest
import torch
import sys


# 在导入模型前检查依赖
def _check_deps():
    """检查可选依赖是否安装"""
    deps = {"ATD": True, "CAMixer": True}

    try:
        import fairscale
    except ImportError:
        deps["ATD"] = False

    try:
        import basicsr
    except ImportError:
        deps["CAMixer"] = False

    return deps


_DEPS = _check_deps()


# 条件导入模型
def _import_models():
    """导入模型模块，跳过不可用的模型"""
    # 先导入不依赖外部包的模块
    from src.models.registry import list_models, get_model_info, MODEL_REGISTRY
    from src.models.base import create_model_result, merge_params

    # 注册基础模型（这些不依赖外部包）
    from src.models import bicubic  # noqa: F401
    from src.models import EDSR  # noqa: F401
    from src.models import RCAN  # noqa: F401
    from src.models import RDN  # noqa: F401
    from src.models import MySR  # noqa: F401
    from src.models import MySRAb  # noqa: F401
    from src.models import SwinIR  # noqa: F401

    # 条件导入 ATD
    if _DEPS["ATD"]:
        try:
            from src.models import ATD  # noqa: F401
        except ImportError:
            _DEPS["ATD"] = False

    # 条件导入 CAMixer（basicsr 和 torchvision 可能有兼容性问题）
    if _DEPS["CAMixer"]:
        try:
            from src.models import CAMixer  # noqa: F401
        except ImportError:
            _DEPS["CAMixer"] = False

    return list_models, get_model_info, create_model_result, merge_params


list_models, get_model_info, create_model_result, merge_params = _import_models()


# 测试配置
BATCH_SIZE = 2
HEIGHT = 32
WIDTH = 32


def is_model_available(model_name: str) -> bool:
    """检查模型是否可用（依赖是否安装）"""
    if model_name == "ATD":
        return _DEPS.get("ATD", False)
    if model_name == "CAMixer":
        return _DEPS.get("CAMixer", False)
    return True


def get_test_models():
    """获取可测试的模型列表"""
    all_models = list_models()
    return [m for m in all_models if is_model_available(m)]


class TestModelRegistry:
    """测试模型注册机制"""

    def test_list_models(self):
        """测试 list_models 返回模型列表"""
        models = list_models()
        assert isinstance(models, list)
        assert len(models) > 0
        # 核心模型应该在列表中
        assert "EDSR" in models
        assert "RCAN" in models
        assert "RDN" in models

    def test_get_model_info(self):
        """测试 get_model_info 返回正确信息"""
        info = get_model_info("EDSR")
        assert info.name == "EDSR"
        assert info.factory is not None
        assert info.default_params is not None
        assert "n_feats" in info.default_params

    def test_get_model_info_invalid(self):
        """测试获取未注册模型抛出异常"""
        with pytest.raises(ValueError, match="未注册"):
            get_model_info("InvalidModel")


class TestModelForward:
    """测试模型前向传播"""

    @pytest.mark.parametrize("model_name", get_test_models())
    def test_model_forward_in_dim_1(self, model_name):
        """测试单通道输入前向传播"""
        info = get_model_info(model_name)

        # Bicubic 返回 None，跳过
        if model_name == "Bicubic":
            pytest.skip("Bicubic has no model")

        params = merge_params(info.default_params or {}, {})
        in_dim = 1
        upscale = 2

        result = info.factory(params, in_dim, upscale)

        assert "model" in result
        assert "model_name" in result

        model = result["model"]
        model.eval()

        # 前向传播
        x = torch.randn(BATCH_SIZE, in_dim, HEIGHT, WIDTH)
        with torch.no_grad():
            y = model(x)

        # 检查输出 shape
        assert y.shape[0] == BATCH_SIZE
        assert y.shape[1] == in_dim
        assert y.shape[2] == HEIGHT * upscale
        assert y.shape[3] == WIDTH * upscale

    @pytest.mark.parametrize("model_name", get_test_models())
    def test_model_forward_in_dim_2(self, model_name):
        """测试双通道输入前向传播（模拟 MWD）"""
        info = get_model_info(model_name)

        if model_name == "Bicubic":
            pytest.skip("Bicubic has no model")

        params = merge_params(info.default_params or {}, {})
        in_dim = 2
        upscale = 2

        result = info.factory(params, in_dim, upscale)
        model = result["model"]
        model.eval()

        # 前向传播
        x = torch.randn(BATCH_SIZE, in_dim, HEIGHT, WIDTH)
        with torch.no_grad():
            y = model(x)

        # 检查输出 shape
        assert y.shape[0] == BATCH_SIZE
        assert y.shape[1] == in_dim
        assert y.shape[2] == HEIGHT * upscale
        assert y.shape[3] == WIDTH * upscale


class TestModelUpscale:
    """测试不同放大倍数"""

    @pytest.mark.parametrize("upscale", [2, 4])
    @pytest.mark.parametrize("model_name", ["EDSR", "RCAN", "RDN", "MySR"])
    def test_model_upscale(self, model_name, upscale):
        """测试不同 upscale 输出 shape 正确"""
        info = get_model_info(model_name)
        params = merge_params(info.default_params or {}, {})
        in_dim = 1

        result = info.factory(params, in_dim, upscale)
        model = result["model"]
        model.eval()

        x = torch.randn(BATCH_SIZE, in_dim, HEIGHT, WIDTH)
        with torch.no_grad():
            y = model(x)

        assert y.shape[2] == HEIGHT * upscale
        assert y.shape[3] == WIDTH * upscale


class TestModelParams:
    """测试模型参数配置"""

    def test_default_params(self):
        """测试默认参数被正确使用"""
        info = get_model_info("EDSR")
        params = merge_params(info.default_params, {})

        result = info.factory(params, 1, 2)
        model_name = result["model_name"]

        # 模型名应包含默认参数
        assert "f64" in model_name  # n_feats=64
        assert "n16" in model_name  # n_resblocks=16

    def test_custom_params(self):
        """测试自定义参数覆盖默认值"""
        info = get_model_info("EDSR")
        custom_params = {"n_feats": 128, "n_resblocks": 8}
        params = merge_params(info.default_params, custom_params)

        result = info.factory(params, 1, 2)
        model_name = result["model_name"]

        # 模型名应包含自定义参数
        assert "f128" in model_name
        assert "n8" in model_name


class TestModelSkip:
    """跳过需要额外依赖的模型测试"""

    @pytest.mark.skipif(
        not is_model_available("ATD"),
        reason="fairscale not installed"
    )
    def test_atd_available(self):
        """ATD 模型可用时运行"""
        info = get_model_info("ATD")
        params = merge_params(info.default_params or {}, {})
        result = info.factory(params, 1, 2)
        assert result["model"] is not None

    @pytest.mark.skipif(
        not is_model_available("CAMixer"),
        reason="basicsr not installed"
    )
    def test_camixer_available(self):
        """CAMixer 模型可用时运行"""
        info = get_model_info("CAMixer")
        params = merge_params(info.default_params or {}, {})
        result = info.factory(params, 1, 2)
        assert result["model"] is not None