from .common import nn
from . import common


class EDSR(nn.Module):
    def __init__(self, scale, n_colors, n_feats, n_resblocks, res_scale=0.2, conv=common.default_conv):
        super(EDSR, self).__init__()

        n_resblocks = n_resblocks
        n_feats = n_feats
        kernel_size = 3
        scale = scale
        act = nn.ReLU(True)

        # define head module
        m_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        return x


# ============== Model Factory ==============

from .registry import register_model
from .base import create_model_result, merge_params

# 默认参数
EDSR_DEFAULT_PARAMS = {
    "n_feats": 64,
    "n_resblocks": 16,
    "res_scale": 0.2
}


@register_model("edsr", default_params=EDSR_DEFAULT_PARAMS)
def create_edsr_single(params: dict, in_dim: int, upscale: int):
    """单参数 EDSR 模型工厂"""
    p = merge_params(EDSR_DEFAULT_PARAMS, params)

    model = EDSR(
        scale=upscale,
        n_colors=in_dim,
        n_feats=p["n_feats"],
        n_resblocks=p["n_resblocks"],
        res_scale=p["res_scale"]
    )

    model_name = f"edsr_f{p['n_feats']}_n{p['n_resblocks']}_x{upscale}"
    return create_model_result(model, model_name)

