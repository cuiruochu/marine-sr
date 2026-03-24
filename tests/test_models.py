"""
模型单元测试

测试所有模型的前向传播、输出 shape、输入通道等。
"""

from importlib.util import find_spec

import pytest
import torch

from src.core import normalize_model_output
from src.models.base import merge_params
from src.models.registry import get_model_info, list_models

# 在导入模型前检查依赖

def _check_deps():
    """检查可选依赖是否安装"""
    return {
        "ATD": find_spec("fairscale") is not None,
        "CAMixer": find_spec("torchvision") is not None,
    }


_DEPS = _check_deps()


# 触发基础模型注册
list_models()


def is_model_available(model_name: str) -> bool:
    """检查模型是否可用（依赖是否安装）"""
    if model_name == "ATD":
        return _DEPS.get("ATD", False)
    if model_name == "CAMixer":
        return _DEPS.get("CAMixer", False)
    return True


def get_test_models():
    """获取可测试的模型列表"""
    return [model_name for model_name in list_models() if is_model_available(model_name)]


class TestModelRegistry:
    """测试模型注册机制"""

    def test_list_models(self):
        """测试 list_models 返回模型列表"""
        models = list_models()
        assert isinstance(models, list)
        assert len(models) > 0
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

        if model_name == "Bicubic":
            pytest.skip("Bicubic has no model")

        params = merge_params(info.default_params or {}, {})
        in_dim = 1
        upscale = 2

        result = info.factory(params, in_dim, upscale)
        model = result["model"]
        model.eval()

        x = torch.randn(2, in_dim, 32, 32)
        with torch.no_grad():
            y = model(x)

        assert y.shape[0] == 2
        assert y.shape[1] == in_dim
        assert y.shape[2] == 64
        assert y.shape[3] == 64


class TestModelBackward:
    """测试模型反向传播"""

    @pytest.mark.parametrize("model_name", get_test_models())
    @pytest.mark.parametrize("in_dim", [1, 2])
    def test_model_backward(self, model_name, in_dim):
        """测试模型可用假数据完成前向和反向传播"""
        if model_name == "Bicubic":
            pytest.skip("Bicubic has no trainable model")

        info = get_model_info(model_name)
        params = merge_params(info.default_params or {}, {})
        upscale = 2

        result = info.factory(params, in_dim, upscale)
        model = result["model"]
        model.train()

        x = torch.randn(1, in_dim, 16, 16)
        target = torch.randn(1, in_dim, 32, 32)

        output = normalize_model_output(model(x))
        y = output.pred
        loss = torch.nn.functional.mse_loss(y, target)
        if output.aux_losses:
            loss = loss + sum(value for value in output.aux_losses.values())
        loss.backward()

        grads = [param.grad for param in model.parameters() if param.requires_grad]
        assert grads, f"{model_name} 没有可训练参数"
        assert any(grad is not None for grad in grads), f"{model_name} 反向传播后没有梯度"
        assert all(grad is None or torch.isfinite(grad).all() for grad in grads), f"{model_name} 梯度包含非有限值"

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

        x = torch.randn(2, in_dim, 32, 32)
        with torch.no_grad():
            y = model(x)

        assert y.shape[0] == 2
        assert y.shape[1] == in_dim
        assert y.shape[2] == 64
        assert y.shape[3] == 64


class TestModelUpscale:
    """测试不同放大倍数"""

    @pytest.mark.parametrize("upscale", [2, 4])
    @pytest.mark.parametrize("model_name", ["EDSR", "RCAN", "RDN", "MySR"])
    def test_model_upscale(self, model_name, upscale):
        """测试不同 upscale 输出 shape 正确"""
        info = get_model_info(model_name)
        params = merge_params(info.default_params or {}, {})

        result = info.factory(params, 1, upscale)
        model = result["model"]
        model.eval()

        x = torch.randn(2, 1, 32, 32)
        with torch.no_grad():
            y = model(x)

        assert y.shape[2] == 32 * upscale
        assert y.shape[3] == 32 * upscale


class TestModelParams:
    """测试模型参数配置"""

    def test_default_params(self):
        """测试默认参数被正确使用"""
        info = get_model_info("EDSR")
        params = merge_params(info.default_params, {})

        result = info.factory(params, 1, 2)
        model_name = result["model_name"]

        assert "f64" in model_name
        assert "n16" in model_name

    def test_custom_params(self):
        """测试自定义参数覆盖默认值"""
        info = get_model_info("EDSR")
        custom_params = {"n_feats": 128, "n_resblocks": 8}
        params = merge_params(info.default_params, custom_params)

        result = info.factory(params, 1, 2)
        model_name = result["model_name"]

        assert "f128" in model_name
        assert "n8" in model_name


class TestModelSkip:
    """跳过需要额外依赖的模型测试"""

    @pytest.mark.skipif(not is_model_available("ATD"), reason="fairscale not installed")
    def test_atd_available(self):
        """ATD 模型可用时运行"""
        info = get_model_info("ATD")
        params = merge_params(info.default_params or {}, {})
        result = info.factory(params, 1, 2)
        assert result["model"] is not None

    @pytest.mark.skipif(not is_model_available("CAMixer"), reason="torchvision not installed")
    def test_camixer_available(self):
        """CAMixer 模型可用时运行"""
        info = get_model_info("CAMixer")
        params = merge_params(info.default_params or {}, {})
        result = info.factory(params, 1, 2)
        assert result["model"] is not None
