from .common import torch, nn
from .MySR import SpatialGatingUnit, Upsample


# Ab1: 删除分支
class ResBlockAb1(nn.Module):
    def __init__(self, features):
        super(ResBlockAb1, self).__init__()
        self.global_body = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(features, features, kernel_size=7, padding=3, groups=features),
            SpatialGatingUnit(features),
        )
        self.fuse_body = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
        )

    def forward(self, x):
        """
            input: (B, C, H, W)
            output: (B, C, H, W)
        """
        shortcut = x
        global_feat = self.global_body(x) + x
        out = self.fuse_body(global_feat) + global_feat
        return out + shortcut


class MySRAb1(nn.Module):
    def __init__(self, upscale=4, in_ch=1, num_features=64, n_blocks=5):
        super(MySRAb1, self).__init__()
        # Shallow Feature Extraction
        self.head = nn.Conv2d(in_ch, num_features, kernel_size=3, padding=1)
        # Deep Feature Extraction
        body = []
        for i in range(n_blocks):
            body.append(ResBlockAb1(num_features))
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


# Ab2: 删除SGU
class ResBlockAb2(nn.Module):
    def __init__(self, features):
        super(ResBlockAb2, self).__init__()
        self.local_body = nn.Sequential(
            nn.Conv2d(features, features // 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(features // 4, features, kernel_size=3, padding=1),
        )
        self.global_body = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(features, features, kernel_size=7, padding=3, groups=features),
            nn.Conv2d(features, features, kernel_size=7, padding=3, groups=features)
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


class MySRAb2(nn.Module):
    def __init__(self, upscale=4, in_ch=1, num_features=64, n_blocks=5):
        super(MySRAb2, self).__init__()
        # Shallow Feature Extraction
        self.head = nn.Conv2d(in_ch, num_features, kernel_size=3, padding=1)
        # Deep Feature Extraction
        body = []
        for i in range(n_blocks):
            body.append(ResBlockAb2(num_features))
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

# MySRAb1 默认参数
MYSRAB1_DEFAULT_PARAMS = {
    "num_features": 76,
    "n_blocks": 5
}

# MySRAb2 默认参数
MYSRAB2_DEFAULT_PARAMS = {
    "num_features": 66,
    "n_blocks": 5
}


@register_model("MySRAb1", default_params=MYSRAB1_DEFAULT_PARAMS)
def create_mysrab1_single(params: dict, in_dim: int, upscale: int):
    """单参数 MySRAb1 模型工厂"""
    p = merge_params(MYSRAB1_DEFAULT_PARAMS, params)

    model = MySRAb1(
        upscale=upscale,
        in_ch=in_dim,
        num_features=p["num_features"],
        n_blocks=p["n_blocks"]
    )

    model_name = f"MySRAb1_f{p['num_features']}_n{p['n_blocks']}_x{upscale}"
    return create_model_result(model, model_name)


@register_model("MySRAb2", default_params=MYSRAB2_DEFAULT_PARAMS)
def create_mysrab2_single(params: dict, in_dim: int, upscale: int):
    """单参数 MySRAb2 模型工厂"""
    p = merge_params(MYSRAB2_DEFAULT_PARAMS, params)

    model = MySRAb2(
        upscale=upscale,
        in_ch=in_dim,
        num_features=p["num_features"],
        n_blocks=p["n_blocks"]
    )

    model_name = f"MySRAb2_f{p['num_features']}_n{p['n_blocks']}_x{upscale}"
    return create_model_result(model, model_name)
