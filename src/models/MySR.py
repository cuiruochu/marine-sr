from .common import torch, nn, math


class SpatialGatingUnit(nn.Module):
    def __init__(self, channels, expand=2):
        super(SpatialGatingUnit, self).__init__()
        hidden_size = int(channels * expand)
        assert hidden_size % 2 == 0, '(channels * expand) % 2 != 0'
        half_hidden_size = int(hidden_size // 2)

        self.norm = nn.LayerNorm(channels)
        self.fc = nn.Linear(channels, hidden_size)
        self.act = nn.GELU()

        self.split_indices = [half_hidden_size, half_hidden_size]
        self.dwconv = nn.Conv2d(half_hidden_size, half_hidden_size, kernel_size=7, padding=7 // 2,
                                groups=half_hidden_size)
        self.fc2 = nn.Conv2d(half_hidden_size, channels, kernel_size=1)

    def forward(self, x):
        """
        x: (B, C, H, W)
        :return: (B, C, H, W)
        """
        shortcut = x

        x = x.permute(0, 2, 3, 1)  # (B, H, W, C) norm -> fc-> Gelu
        x = self.norm(x)
        x = self.fc(x)
        x = self.act(x)
        x = x.permute(0, 3, 1, 2)

        g, i = torch.split(x, self.split_indices, dim=1)  # (B, hidden, H, W) -> split -> (B, hidden/2, H, W)
        x = self.dwconv(g) * i

        x = self.fc2(x)  # (B, C, H, W)
        return x + shortcut


class ResBlock(nn.Module):
    def __init__(self, features):
        super(ResBlock, self).__init__()
        self.local_body = nn.Sequential(
            nn.Conv2d(features, features // 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(features // 4, features, kernel_size=3, padding=1),
        )
        self.global_body = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(features, features, kernel_size=7, padding=3, groups=features),
            SpatialGatingUnit(features),
        )
        self.fuse_body = nn.Sequential(
            nn.Conv2d(2 * features, features, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
        )
        self.fuse_res = nn.Conv2d(2 * features, features, kernel_size=1)

    def forward(self, x):
        """
            input: (B, C, H, W)
            output: (B, C, H, W)
        """
        shortcut = x
        local_feat = self.local_body(x) + x
        global_feat = self.global_body(x) + x
        feat = torch.cat([local_feat, global_feat], dim=1)
        out = self.fuse_body(feat) + self.fuse_res(feat)
        return out + shortcut


class Upsample(nn.Module):
    def __init__(self, n_channels, scale, groups=1):
        super(Upsample, self).__init__()
        self.body = nn.ModuleList()
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                self.body.append(nn.Conv2d(n_channels, 4 * n_channels, 3, 1, 1, groups=groups))
                self.body.append(nn.PixelShuffle(2))
        elif scale == 3:
            self.body.append(nn.Conv2d(n_channels, 9 * n_channels, 3, 1, 1, groups=groups))
            self.body.append(nn.PixelShuffle(3))

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        return out


class MySR(nn.Module):
    def __init__(self, upscale=4, in_ch=1, num_features=64, n_blocks=5):
        super(MySR, self).__init__()
        # Shallow Feature Extraction
        self.head = nn.Conv2d(in_ch, num_features, kernel_size=3, padding=1)
        # Deep Feature Extraction
        body = []
        for i in range(n_blocks):
            body.append(ResBlock(num_features))
        body.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
        self.body = nn.Sequential(*body)
        # Reconstruction
        self.upsample = nn.Sequential(
            Upsample(num_features, upscale),
            nn.Conv2d(num_features, in_ch, kernel_size=3, padding=1)
        )

    def forward(self, x):
        """
            input:(B, 1, H, W)
            output:(B, 1, upscale*H, upscale*W)
        """
        # Shallow Feature Extraction
        x = self.head(x)
        shortcut = x
        # Deep Feature Extraction
        x = self.body(x)
        # Reconstruction
        out = self.upsample(x + shortcut)
        return out


# ============== Model Factory ==============

from .registry import register_model
from .base import create_model_result, merge_params

# 默认参数
MYSR_DEFAULT_PARAMS = {
    "num_features": 64,
    "n_blocks": 5
}


@register_model("MySR", default_params=MYSR_DEFAULT_PARAMS)
def create_mysr_single(params: dict, in_dim: int, upscale: int):
    """单参数 MySR 模型工厂"""
    p = merge_params(MYSR_DEFAULT_PARAMS, params)

    model = MySR(
        upscale=upscale,
        in_ch=in_dim,
        num_features=p["num_features"],
        n_blocks=p["n_blocks"]
    )

    model_name = f"MySR_f{p['num_features']}_n{p['n_blocks']}_x{upscale}"
    return create_model_result(model, model_name)
