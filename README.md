# 项目架构问题记录

本文档记录当前项目存在的架构问题，用于指导后续重构。

---

## 一、可扩展性问题

### 1. 拓展模型困难

**现状：** 使用 if-elif 链，每添加一个模型需要修改 `src/models/__init__.py`

```python
# src/models/__init__.py
def _create_single_model(model_config):
    if name == "Bicubic":
        ...
    elif name == "CAMixer":
        ...
    elif name == "ATD":
        ...
    # 添加新模型需要在这里加 elif
```

**问题：**
- 违反开闭原则
- 代码冗长，难以维护
- 模型超参数硬编码在 `__init__.py`

**解决方案：** 引入模型注册机制

```python
# src/models/registry.py
MODEL_REGISTRY = {}

def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

# src/models/EDSR.py
@register_model("EDSR")
def create_edsr_single(params, in_dim, upscale):
    ...
```

**实际修改（已完成）：**
- 新建 `src/models/registry.py`：提供 `@register_model` 和 `@register_unified_model` 装饰器
- 新建 `src/models/base.py`：提供 `create_model_result()` 和 `merge_params()` 工具函数
- 新建 `src/models/bicubic.py`：Bicubic 基线模型工厂
- 修改所有模型文件（EDSR, RCAN, RDN, MySR, MySRAb, SwinIR, ATD, CAMixer）：添加工厂函数和装饰器
- 重写 `src/models/__init__.py`：使用 `MODEL_REGISTRY` 查找模型

添加新模型只需在模型文件中加 `@register_model("xxx")`，无需修改 `__init__.py`。

---

### 2. 拓展 Dataset 困难

**现状：** 参数类型硬编码，if-elif 判断参数是否合法

```python
# src/datasets/dataset.py
if marine_param in ["wind", "mwd", "mwp", "swh"]:
    train_set = MarineTrainSet(...)
else:
    raise NotImplementedError
```

**分析：**
- 所有参数用的都是同一个 `MarineTrainSet` 类
- if-elif 只是判断参数是否合法，不是选择不同类
- 不需要注册机制

**解决方案：** 参数类型放到配置文件

```yaml
dataset:
  name: marine
  params:
    marine_param: wind  # 配置文件指定，无需代码判断
```

如果未来需要支持不同数据集类，再引入注册机制。

**实际修改（已完成）：**
- 更新 `configs/default.yaml`：添加 `marine_params.valid` 和 `marine_params.channels` 配置
- 更新 `configs/loader.py`：`TrainConfig` 和 `TestConfig` 添加 `_validate()` 方法，在配置加载时验证参数
- 修改 `src/datasets/dataset.py`：移除 if-elif 参数验证，添加 `_is_mwd_param()` 辅助函数

---

### 3. 训练设置配置分散

**现状：** 配置分散在三处，未集中管理

| 配置项 | 当前位置 | 问题 |
|--------|----------|------|
| lr, batch_size, epoches | `scripts/train.py` | 硬编码，不灵活 |
| 模型超参数 (n_feats, n_blocks) | `src/models/__init__.py` | 与模型定义分离 |
| 数据路径 | `configs/*.py` | 绝对路径，跨环境不可用 |

**解决方案：** 使用 YAML 配置文件集中管理

---

### 4. 无配置文件驱动训练

**现状：** 使用 Python dataclass 硬编码，修改配置需要改源码

**解决方案：** 使用 YAML 配置文件

---

### 5. 无配置文件驱动测试

**现状：** 同上，测试配置也硬编码在 `configs/test.yaml`

**解决方案：** 使用 YAML 配置文件

---

**问题 3-5 统一解决方案：** 一个 YAML 文件集中管理所有配置

```yaml
# configs/train.yaml
model:
  name: EDSR
  params: { n_feats: 64, n_resblocks: 16 }

dataset:
  name: marine
  params: { marine_param: wind, upscale: 2, lr_patch_size: 30 }

train:
  epochs: 200
  batch_size: 16
  lr: 2e-4

paths:
  train_root: /data/marine/Train/wind
  checkpoint_dir: ./checkpoints
```

---

## 二、可测试性问题

### 6. 缺少单元测试

**现状：** 项目无任何测试代码

**问题：**
- 无法验证模型实现正确性
- 重构时无法保证不破坏现有功能
- 无法验证数据加载逻辑

**解决方案：** 添加测试目录

```
tests/
├── test_models.py      # 确保模型能跑
└── test_datasets.py    # 确保数据能加载
```

| 文件 | 测试点 | 目的 |
|------|--------|------|
| test_models.py | forward、upscale、in_dim | 确保模型能跑、shape 对 |
| test_datasets.py | getitem、normalization、mwd_encoding | 确保数据加载对、预处理对 |

**test_models.py：**

```python
def test_model_forward():
    """所有模型能否正常前向传播"""

def test_model_upscale():
    """不同 upscale (2, 4) 输出 shape 是否正确"""

def test_model_in_dim():
    """不同输入通道 (1 或 2) 是否正常"""
```

**test_datasets.py：**

```python
def test_dataset_getitem():
    """__getitem__ 返回的 lr, hr shape 是否正确"""

def test_normalization():
    """归一化是否正确"""

def test_mwd_encoding():
    """mwd 的 cos/sin 编码是否正确"""
```

---

## 三、可复现性问题

### 7. 缺少依赖管理

**现状：** 无 `requirements.txt` 或 `pyproject.toml`

**问题：**
- 换环境无法复现
- 依赖版本不明确
- 别人无法快速上手

**解决方案：** 使用 uv 管理依赖

```
pyproject.toml      # 项目元数据和依赖
uv.lock             # 锁定版本（uv 自动生成）
```

`pyproject.toml` 示例：

```toml
[project]
name = "marine-sr"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
    "numpy",
    "tqdm",
    "wandb",
    "pyyaml",
    "einops",
    "timm",
]

[project.optional-dependencies]
dev = ["pytest", "ruff"]
all = ["fairscale", "basicsr"]  # 可选模型依赖
```

**实际修改（已完成）：**
- 新建 `pyproject.toml`：定义项目元数据和依赖

---

### 8. 路径硬编码

**现状：** 数据路径使用绝对路径，相对路径依赖工作目录

```yaml
# configs/train.yaml
train_root: /data/marine/Train/wind  # 需要用户修改
```

**问题：**
- 跨机器/跨环境不可用
- 相对路径基准不明确

**解决方案：** 两层解决

| 问题 | 解决方式 |
|------|----------|
| 绝对路径硬编码 | YAML 配置文件 |
| 相对路径基准 | 代码中定义 PROJECT_ROOT |

**实现方式：**

1. 定义项目根目录（解决相对路径基准）

```python
# src/utils/path.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
```

2. 配置文件中直接写路径

```yaml
# configs/train.yaml
paths:
  data_root: /data/marine        # 绝对路径
  checkpoint_dir: ./checkpoints  # 相对路径（相对于项目根目录）
```

3. 代码中解析路径

```python
def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
```

---

## 四、可追踪性问题

### 9. 实验与配置分离

**现状：** 使用 wandb offline，配置与 checkpoint 未关联

**问题：**
- 不知道哪个 checkpoint 对应哪个配置
- 实验多了难以对比和复现

**解决方案：** 每次实验生成独立目录，把相关文件放一起

```
experiments/
├── exp_20240315_edsr_wind_x2/
│   ├── config.yaml        # 这次实验的配置
│   ├── train_log.json     # 训练曲线
│   └── checkpoints/
│       ├── epoch_10.pth
│       └── epoch_20.pth
└── exp_20240316_rcan_mwp_x4/
    ├── config.yaml
    ├── train_log.json
    └── checkpoints/
```

**同时：** 用实验目录名作为 wandb 的实验名，保持关联

```python
# 训练脚本中
exp_name = "exp_20240315_edsr_wind_x2"
wandb.init(
    project="Marine-Parameters-SupervisedSR",
    name=exp_name,  # 与目录名一致
)
```

这样 wandb 中的实验名与本地目录名对应，方便查找

---

## 五、代码规范问题

### 10. 目录命名不一致

**现状：** 已完成重命名，统一使用复数

```
src/
├── models/     ✓
├── trainers/   ✓
├── testers/    ✓
├── datasets/   ✓
└── utils/      ✓

configs/        ✓
```

---

### 11. 使用 print 而非日志系统

**现状：** 所有输出使用 print

```python
print("当前训练 unified 模型!!!")
print(f"Total trainable parameters: {total_params}")
```

**问题：**
- 无法控制日志级别
- 无法持久化日志
- 无法追踪问题

**解决方案：** 使用 Python logging 模块

```python
import logging

logger = logging.getLogger(__name__)

# 使用
logger.info("当前训练 unified 模型")
logger.info(f"Total trainable parameters: {total_params}")
```

---

## 六、总结

| 问题类别 | 问题数量 | 优先级 |
|----------|----------|--------|
| 可扩展性 | 5 | 高 |
| 可测试性 | 1 | 高 |
| 可复现性 | 2 | 中 |
| 可追踪性 | 1 | 中 |
| 代码规范 | 2 | 低 |

**建议重构顺序：**

```
第一阶段：基础设施
1. 目录重命名（问题 10）── 先改，后续不用改 import
2. PROJECT_ROOT（问题 8 的一部分）── 路径解析基础
3. YAML 配置文件系统（问题 3-5, 8）── 核心基础

第二阶段：核心重构
4. 模型注册机制（问题 1）── 依赖配置文件
5. Dataset 参数配置化（问题 2）── 依赖配置文件

第三阶段：验证与完善
6. 单元测试（问题 6）── 重构完成后再写

第四阶段：增强功能
7. 实验管理（问题 9）── 依赖配置文件
8. 日志系统（问题 11）── 独立
```

**依赖关系：**

| 问题 | 依赖 |
|------|------|
| 模型注册（问题 1） | 需要配置文件提供模型名和参数 |
| Dataset 参数（问题 2） | 需要配置文件指定参数 |
| 实验管理（问题 9） | 需要配置文件 + PROJECT_ROOT |
| 单元测试（问题 6） | 应在代码稳定后写 |

---

## 七、重构进度

### ✅ 第一阶段：基础设施（已完成）

| 步骤 | 状态 | 说明 |
|------|------|------|
| 1. 目录重命名 | ✅ 完成 | config→configs, dataset→datasets, tester→testers |
| 2. PROJECT_ROOT | ✅ 完成 | 新建 src/utils/path.py |
| 3. YAML 配置系统 | ✅ 完成 | 新建 configs/train.yaml, test.yaml, loader.py |
| 4. 源码移入 src | ✅ 完成 | models, datasets, trainers, testers, utils → src/ |

**目录结构：**

```
第三章/
├── src/                    # 源码
│   ├── models/
│   ├── datasets/
│   ├── trainers/
│   ├── testers/
│   └── utils/
├── configs/                # 配置文件
│   ├── default.yaml
│   ├── train.yaml
│   ├── test.yaml
│   └── loader.py
├── scripts/               # 入口脚本
│   ├── train.py
│   ├── eval.py
│   └── find_lr.py
├── checkpoints/            # 模型权重（自动创建）
├── tests/                  # 测试（待添加）
└── experiments/            # 实验输出（待添加）
```

### ✅ 第二阶段：核心重构（已完成）

| 步骤 | 状态 | 说明 |
|------|------|------|
| 4. 模型注册机制 | ✅ 完成 | 新建 registry.py, base.py, bicubic.py；各模型添加工厂函数 |
| 5. Dataset 参数配置化 | ✅ 完成 | 配置验证移至 TrainConfig/TestConfig；移除 if-elif |

**新增文件：**
- `src/models/registry.py` - 模型注册机制
- `src/models/base.py` - 工具函数
- `src/models/bicubic.py` - Bicubic 基线模型

**修改文件：**
- `src/models/*.py` - 各模型添加 `@register_model` 装饰器和工厂函数
- `src/models/__init__.py` - 使用注册表查找模型
- `configs/default.yaml` - 添加 `marine_params` 配置
- `configs/loader.py` - 添加 `_validate()` 验证方法
- `src/datasets/dataset.py` - 移除 if-elif，添加 `_is_mwd_param()` 辅助函数

### ⏳ 第三阶段：验证与完善（待开始）

| 步骤 | 状态 |
|------|------|
| 6. 单元测试 | ⏳ 待开始 |

### ⏳ 第四阶段：增强功能（待开始）

| 步骤 | 状态 |
|------|------|
| 7. 实验管理 | ⏳ 待开始 |
| 8. 日志系统 | ⏳ 待开始 |