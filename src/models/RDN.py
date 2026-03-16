from .common import torch, nn


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


class RDN(nn.Module):
    def __init__(self, n_colors, scale, n_features, n_blocks=8, layers=4):
        super(RDN, self).__init__()
        r = scale
        G0 = n_features
        kSize = 3

        # number of RDB blocks, conv layers, out channels
        self.D = n_blocks
        C = layers
        G = 32

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(n_colors, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0=G0, growRate=G, nConvLayers=C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

        # Up-sampling net
        if r == 2 or r == 3:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, G * r * r, kSize, padding=(kSize - 1) // 2, stride=1),
                nn.PixelShuffle(r),
                nn.Conv2d(G, n_colors, kSize, padding=(kSize - 1) // 2, stride=1)
            ])
        elif r == 4:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, G * 4, kSize, padding=(kSize - 1) // 2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, G * 4, kSize, padding=(kSize - 1) // 2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, n_colors, kSize, padding=(kSize - 1) // 2, stride=1)
            ])
        else:
            raise ValueError("scale must be 2 or 3 or 4.")

    def forward(self, x):
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f__1

        return self.UPNet(x)


# ============== Model Factory ==============

from .registry import register_model, register_unified_model
from .base import create_model_result, merge_params
from .MWDEncoder import Encoder, Decoder

# 默认参数
RDN_DEFAULT_PARAMS = {
    "n_features": 64,
    "n_blocks": 6,
    "layers": 4
}


@register_model("RDN", default_params=RDN_DEFAULT_PARAMS)
def create_rdn_single(params: dict, in_dim: int, upscale: int):
    """单参数 RDN 模型工厂"""
    p = merge_params(RDN_DEFAULT_PARAMS, params)

    model = RDN(
        n_colors=in_dim,
        scale=upscale,
        n_features=p["n_features"],
        n_blocks=p["n_blocks"],
        layers=p["layers"]
    )

    model_name = f"RDN_f{p['n_features']}_n{p['n_blocks'] * p['layers']}_x{upscale}"
    return create_model_result(model, model_name)


@register_unified_model("RDN", default_params=RDN_DEFAULT_PARAMS)
def create_rdn_unified(params: dict, upscale: int):
    """统一 RDN 模型工厂（支持多参数训练）"""
    p = merge_params(RDN_DEFAULT_PARAMS, params)
    n_features = p["n_features"]

    mwd_encoder = Encoder(in_ch=2, hidden_dim=n_features)
    other_encoder = Encoder(in_ch=1, hidden_dim=n_features)
    mwd_decoder = Decoder(hidden_dim=n_features, out_ch=2)
    other_decoder = Decoder(hidden_dim=n_features, out_ch=1)

    model = RDN(
        n_colors=n_features,
        scale=upscale,
        n_features=n_features,
        n_blocks=p["n_blocks"],
        layers=p["layers"]
    )

    model_name = f"RDN_f{n_features}_n{p['n_blocks'] * p['layers']}_x{upscale}"
    return create_model_result(
        model, model_name,
        mwd_encoder, other_encoder, mwd_decoder, other_decoder
    )

