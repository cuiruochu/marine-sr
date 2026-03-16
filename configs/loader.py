import yaml
import os
from dataclasses import dataclass, field
from typing import List

from src.utils.path import resolve_path


def load_yaml(yaml_path: str) -> dict:
    """加载 YAML 配置文件"""
    path = resolve_path(yaml_path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_default_config() -> dict:
    """加载 default.yaml 配置"""
    return load_yaml("configs/default.yaml")


@dataclass
class TrainConfig:
    """训练配置"""
    # 原始数据
    raw: dict = field(default_factory=dict, repr=False)

    # 模型
    model_name: str = ""
    model_params: dict = field(default_factory=dict)

    # 数据集
    marine_param: str = ""
    upscale: int = 2
    lr_patch_size: int = 30

    # 训练
    epochs: int = 200
    batch_size: int = 16
    lr: float = 2e-4

    # 路径
    train_root: str = ""
    val_root: str = ""
    checkpoint_dir: str = ""
    eval_mask: str = ""

    # 派生字段
    unified: bool = False
    mean: list = field(default_factory=list)
    std: list = field(default_factory=list)
    in_dim: int = 1
    pth_save_path: str = ""

    # 有效参数列表（从 default.yaml 加载）
    _valid_params: List[str] = field(default_factory=lambda: ["wind", "mwd", "mwp", "swh"], repr=False)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainConfig":
        raw = load_yaml(yaml_path)
        config = cls(raw=raw)

        # 加载默认配置
        default = load_default_config()

        # 解析模型
        model = raw.get("model", {})
        config.model_name = model.get("name", "")
        config.model_params = model.get("params", {})

        # 解析数据集
        dataset = raw.get("dataset", {})
        config.marine_param = dataset.get("marine_param", "")
        config.upscale = dataset.get("upscale", 2)
        config.lr_patch_size = dataset.get("lr_patch_size", 30)

        # 解析训练
        train = raw.get("train", {})
        config.epochs = train.get("epochs", 200)
        config.batch_size = train.get("batch_size", 16)
        config.lr = train.get("lr", 2e-4)

        # 解析路径
        paths = raw.get("paths", {})
        config.train_root = paths.get("train_root", "")
        config.val_root = paths.get("val_root", "")
        config.checkpoint_dir = paths.get("checkpoint_dir", "")
        config.eval_mask = paths.get("eval_mask", "")

        # 从默认配置加载有效参数列表
        marine_config = default.get("marine_params", {})
        config._valid_params = marine_config.get("valid", ["wind", "mwd", "mwp", "swh"])

        # 验证并初始化派生字段
        config._validate()
        config._init_derived_fields(default)

        return config

    def _validate(self):
        """验证配置值"""
        # 验证 marine_param
        if self.marine_param not in self._valid_params:
            raise ValueError(
                f"无效的 marine_param '{self.marine_param}'。"
                f"有效选项: {self._valid_params}"
            )

        # 验证 upscale
        if self.upscale not in [2, 4]:
            raise ValueError(f"无效的 upscale '{self.upscale}'。有效选项: [2, 4]")

        # 验证模型名称
        if not self.model_name:
            raise ValueError("model.name 是必需的")

    def _init_derived_fields(self, default_config: dict):
        """初始化派生字段"""
        # 从配置获取通道数
        marine_channels = default_config.get("marine_params", {}).get("channels", {})
        self.in_dim = marine_channels.get(self.marine_param, 1)

        # 回退：如果配置中没有，使用默认规则
        if self.in_dim == 1 and self.marine_param == "mwd":
            self.in_dim = 2

        self._load_normalize(default_config)
        self._init_pth_save_path()

    def _load_normalize(self, default_config: dict):
        """从配置加载归一化参数"""
        norm = default_config.get("normalize", {}).get(self.marine_param, {})
        self.mean = norm.get("mean", [0])
        self.std = norm.get("std", [1])

    def _init_pth_save_path(self):
        """初始化 checkpoint 保存路径"""
        checkpoint_dir = self.checkpoint_dir or "./checkpoints"
        self.pth_save_path = os.path.join(
            checkpoint_dir,
            self.model_name,
            self.marine_param,
            f"x{self.upscale}"
        )
        os.makedirs(self.pth_save_path, exist_ok=True)


@dataclass
class TestConfig:
    """测试配置"""
    # 原始数据
    raw: dict = field(default_factory=dict, repr=False)

    # 模型
    model_name: str = ""

    # 数据集
    marine_param: str = ""
    upscale: int = 2

    # 路径
    test_root: str = ""
    checkpoint: str = ""
    eval_mask: str = ""

    # 派生字段
    unified: bool = False
    mean: list = field(default_factory=list)
    std: list = field(default_factory=list)
    in_dim: int = 1

    # 有效参数列表
    _valid_params: List[str] = field(default_factory=lambda: ["wind", "mwd", "mwp", "swh"], repr=False)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TestConfig":
        raw = load_yaml(yaml_path)
        config = cls(raw=raw)

        # 加载默认配置
        default = load_default_config()

        # 解析模型
        model = raw.get("model", {})
        config.model_name = model.get("name", "")

        # 解析数据集
        dataset = raw.get("dataset", {})
        config.marine_param = dataset.get("marine_param", "")
        config.upscale = dataset.get("upscale", 2)

        # 解析路径
        paths = raw.get("paths", {})
        config.test_root = paths.get("test_root", "")
        config.checkpoint = paths.get("checkpoint", "")
        config.eval_mask = paths.get("eval_mask", "")

        # 从默认配置加载有效参数列表
        marine_config = default.get("marine_params", {})
        config._valid_params = marine_config.get("valid", ["wind", "mwd", "mwp", "swh"])

        # 验证并初始化
        config._validate()
        config._init_derived_fields(default)

        return config

    def _validate(self):
        """验证配置值"""
        if self.marine_param not in self._valid_params:
            raise ValueError(
                f"无效的 marine_param '{self.marine_param}'。"
                f"有效选项: {self._valid_params}"
            )

        if self.upscale not in [2, 4]:
            raise ValueError(f"无效的 upscale '{self.upscale}'。有效选项: [2, 4]")

        if not self.model_name:
            raise ValueError("model.name 是必需的")

    def _init_derived_fields(self, default_config: dict):
        """初始化派生字段"""
        marine_channels = default_config.get("marine_params", {}).get("channels", {})
        self.in_dim = marine_channels.get(self.marine_param, 1)

        if self.in_dim == 1 and self.marine_param == "mwd":
            self.in_dim = 2

        self._load_normalize(default_config)

    def _load_normalize(self, default_config: dict):
        """加载归一化参数"""
        norm = default_config.get("normalize", {}).get(self.marine_param, {})
        self.mean = norm.get("mean", [0])
        self.std = norm.get("std", [1])