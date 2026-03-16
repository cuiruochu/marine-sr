from .common import nn
from . import common


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


## Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    def __init__(self, scale, in_ch, out_ch, n_feats, n_resgroups, n_resblocks, reduction, conv=common.default_conv):
        super(RCAN, self).__init__()

        n_resgroups = n_resgroups
        n_resblocks = n_resblocks
        n_feats = n_feats
        kernel_size = 3
        reduction = reduction
        scale = scale

        # define head module
        modules_head = [conv(in_ch, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, out_ch, kernel_size)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        return x


# ============== Model Factory ==============

from .registry import register_model, register_unified_model
from .base import create_model_result, merge_params
from .MWDEncoder import Encoder, Decoder

# 默认参数
RCAN_DEFAULT_PARAMS = {
    "n_feats": 64,
    "n_resgroups": 3,
    "n_resblocks": 4,
    "reduction": 16
}


@register_model("RCAN", default_params=RCAN_DEFAULT_PARAMS)
def create_rcan_single(params: dict, in_dim: int, upscale: int):
    """单参数 RCAN 模型工厂"""
    p = merge_params(RCAN_DEFAULT_PARAMS, params)

    model = RCAN(
        scale=upscale,
        in_ch=in_dim,
        out_ch=in_dim,
        n_feats=p["n_feats"],
        n_resgroups=p["n_resgroups"],
        n_resblocks=p["n_resblocks"],
        reduction=p["reduction"]
    )

    model_name = f"RCAN_f{p['n_feats']}_n{p['n_resgroups'] * p['n_resblocks']}_x{upscale}"
    return create_model_result(model, model_name)


@register_unified_model("RCAN", default_params=RCAN_DEFAULT_PARAMS)
def create_rcan_unified(params: dict, upscale: int):
    """统一 RCAN 模型工厂（支持多参数训练）"""
    p = merge_params(RCAN_DEFAULT_PARAMS, params)
    n_feats = p["n_feats"]

    mwd_encoder = Encoder(in_ch=2, hidden_dim=n_feats)
    other_encoder = Encoder(in_ch=1, hidden_dim=n_feats)
    mwd_decoder = Decoder(hidden_dim=n_feats, out_ch=2)
    other_decoder = Decoder(hidden_dim=n_feats, out_ch=1)

    model = RCAN(
        upscale, n_feats, n_feats, n_feats,
        p["n_resgroups"], p["n_resblocks"], p["reduction"]
    )

    model_name = f"RCAN_f{n_feats}_n{p['n_resgroups'] * p['n_resblocks']}_x{upscale}"
    return create_model_result(
        model, model_name,
        mwd_encoder, other_encoder, mwd_decoder, other_decoder
    )
