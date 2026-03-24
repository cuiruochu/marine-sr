# Marine Parameter Super-Resolution

基于 PyTorch + Hydra 的海洋参数超分辨率项目。当前支持：

- 训练
- 配对评估
- 离线推理
- 按 `epoch` 恢复的断点续训
- 多模型统一接入与辅助损失训练

README 的目标不是记录历史，而是回答 3 个问题：

1. 怎么快速跑起来
2. 配置为什么这样组织
3. 新模型怎么接进来

## 快速开始

### 1. 安装依赖

项目当前采用“两种后端二选一”的安装方式。

CPU 环境：

```bash
uv sync --extra cpu
```

CUDA 12.8 环境：

```bash
uv sync --extra cu128
```

两者都会安装当前项目的默认依赖，区别只在于 `torch` / `torchvision` 的后端来源。

不要同时执行：

```bash
uv sync --extra cpu --extra cu128
```

验证环境：

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

- CPU 环境下，`torch.version.cuda` 通常是 `None`
- CUDA 环境下，`torch.version.cuda` 应非空，且 `torch.cuda.is_available()` 应为 `True`

### 2. 选择模型

修改 `configs/models/*.yaml`，例如：

```yaml
# configs/models/edsr.yaml
name: edsr
params:
  n_feats: 64
  n_resblocks: 16
  res_scale: 0.2
```

### 3. 配好数据路径后直接运行

训练：

```bash
uv run python scripts/train.py
```

评估：

```bash
uv run python scripts/evaluate.py evaluate.checkpoint=./checkpoints/.../best.pth
```

推理：

```bash
uv run python scripts/infer.py infer.checkpoint=./checkpoints/.../best.pth
```

### 4. 串行实验队列

如果你需要“一个实验结束立刻跑下一个，失败也继续下一个”，可以使用队列执行器：

```bash
uv run python scripts/run_queue.py --queue jobs/queue.txt
```

当前 `jobs/queue.txt` 是训练队列模板，覆盖：

- 全部内置模型
- `wind` / `mwd` / `mwp` / `swh` 四个数据集
- 其中 `mwd` 按 2 通道处理，其余数据集按 1 通道处理

队列中的命令优先通过 `--config-name <dataset>/train_x{scale}` 选择数据集配置，例如：

```bash
uv run python scripts/train.py --config-name wind/train_x2 models=edsr
uv run python scripts/train.py --config-name mwd/train_x4 models=swinir
```

也就是说，数据路径、`upscale`、`data_norm`、默认 `lr_patch_size` 已经放进对应 YAML，命令行通常只需要覆盖少量实验变量，例如：

- `models=...`
- `models.in_dim=...`
- `train.lr=...`
- `resume.checkpoint=...`

只检查任务而不真正执行：

```bash
uv run python scripts/run_queue.py --queue jobs/queue.txt --dry-run
```

### 5. 数据预处理

如果你手头只有原始 HR `.npy` 目录，可以先用预处理脚本统一生成训练用的 `HR/LR` 配对数据：

```bash
uv run python scripts/preprocess_dataset.py --hr-input-dir ./raw_hr --hr-output-dir ./data/wind/train/x2/hr --lr-output-dir ./data/wind/train/x2/lr --scale 2
```

对于 `mwd` 数据集，建议先进行 `cos/sin` 双通道编码，再做下采样预处理。原因是 `mwd` 表示角度，训练时若使用 `MSE` 一类逐点重建损失，直接回归角度值会把 `0°` 和 `360°` 误当成相距很远的数值。推荐流程是：

```bash
uv run python scripts/encode_mwd_cos_sin.py --input-dir ./raw_mwd_hr --output-dir ./encoded_mwd_hr
uv run python scripts/preprocess_dataset.py --hr-input-dir ./encoded_mwd_hr --hr-output-dir ./data/mwd/train/x2/hr --lr-output-dir ./data/mwd/train/x2/lr --scale 2
```

`scripts/encode_mwd_cos_sin.py` 会：

- 读取目录下全部 `mwd` `.npy`
- 若样本中包含 `None`，先补成 `0`
- 统一转成 `float32`
- 按角度值输出 `2×H×W` 的 `cos/sin` 双通道编码
- 保留原文件名保存到输出目录

如果你会对多个数据集分别做预处理，建议显式指定统计文件路径，避免默认的 `stats.json` 被后一次运行覆盖，例如：

```bash
uv run python scripts/preprocess_dataset.py --hr-input-dir ./raw_hr --hr-output-dir ./data/wind/train/x2/hr --lr-output-dir ./data/wind/train/x2/lr --scale 2 --stats-path ./data/wind/train/stats_wind_x2.json
```

这个脚本会：

- 读取输入目录下全部 `.npy`
- 统一保存为 `float32` 的 `C×H×W`
- 若尺寸不能被 `scale` 整除，则裁掉右侧和下侧多余像素
- 使用 torchvision 的 bicubic 插值生成 LR
- 按全体处理后 HR 样本统计 `mean/std`
- 输出 `stats.json`（或通过 `--stats-path` 指定的文件）

约束：

- 原始 HR 输入目录、处理后 HR 输出目录、LR 输出目录必须互不相同
- 原图中若包含 `None`，默认填成 `0`
- 单通道输入可以是 `H×W`
- 多通道输入可以是 `C×H×W`

## 项目结构

```text
src/
├── app/         # 配置解析、对象构建、训练/评估/推理入口
├── callbacks/   # checkpoint、日志、指标、结果保存
├── core/        # Engine、Evaluator、ModelOutput
├── datamodules/ # dataloader、sampler、运行前数据校验
├── datasets/    # 训练 / 评估 / 推理数据集
├── losses/      # 损失函数注册表与实现
├── models/      # 模型注册表与实现
├── optim/       # 优化器与调度器工厂
└── utils/       # 日志、路径、随机数状态、数据文件工具

configs/
├── models/      # 共享模型配置
├── train.yaml   # 通用训练配置
├── evaluate.yaml# 通用配对评估配置
├── infer.yaml   # 通用离线推理配置
├── wind/        # wind 数据集按 x2/x4 拆开的任务配置
├── mwd/         # mwd 数据集按 x2/x4 拆开的任务配置
├── mwp/         # mwp 数据集按 x2/x4 拆开的任务配置
└── swh/         # swh 数据集按 x2/x4 拆开的任务配置

scripts/
├── train.py
├── evaluate.py
├── infer.py
├── run_queue.py
└── preprocess_dataset.py
```

## 配置设计

这个项目把配置拆成“模型”和“任务场景”两层。

### 共享部分：模型配置

`configs/models/*.yaml` 只描述模型本身：

- `name`
- `params`

模型配置会被训练、评估、推理共同复用。

### 不共享部分：任务配置

三种任务的字段差异很大，所以拆成三份主配置：

- `configs/train.yaml`
- `configs/evaluate.yaml`
- `configs/infer.yaml`

这样做的好处是：

- 训练不会混入测试字段
- 推理不会要求提供 HR
- 配置更短，更容易检查
- 运行前校验更明确

在这三份通用配置之外，项目还提供了按数据集和放大倍数拆开的任务配置，例如：

- `configs/wind/train_x2.yaml`
- `configs/wind/evaluate_x4.yaml`
- `configs/mwd/infer_x2.yaml`

这些 YAML 会把数据路径、`upscale`、`data_norm`、`models.in_dim` 等固定信息直接写死，方便队列执行和批量实验。

## 训练

训练配置只关心这几类信息：

- 数据路径
- `upscale`
- patch / 归一化
- `models.in_dim`
- 优化器
- 学习率调度器
- 损失函数
- checkpoint
- resume

### 最小训练 YAML

```yaml
defaults:
  - models: edsr
  - _self_

models:
  in_dim: 1

seed: 0
upscale: 2

data_norm:
  mean: [4.565732]
  std: [3.1361]

dataset:
  name: wind
  lr_patch_size: 30
  train_lr_root: ./data/wind/train/x2/lr
  train_hr_root: ./data/wind/train/x2/hr
  val_lr_root: ./data/wind/test/x2/lr
  val_hr_root: ./data/wind/test/x2/hr
  max_sample: false # false=全量；正整数=只取前 N 个验证样本

train:
  epochs: 200
  batch_size: 16
  lr: 2e-4
  optimizer:
    name: adamw
    params:
      betas: [0.9, 0.999]
      eps: 0.00000001
      weight_decay: 0.0001
  scheduler:
    name: cosine
    params:
      T_max: ${train.epochs} # 按 epoch 调度：每个 epoch 调一次 scheduler.step()
      eta_min: 0.000001
  loss:
    name: l1
    params: {}
```

### 启动命令

```bash
uv run python scripts/train.py
```

常用 override 示例：

```bash
uv run python scripts/train.py models=camixer train.epochs=50 train.batch_size=8
uv run python scripts/train.py dataset.train_lr_root=./data/... dataset.train_hr_root=./data/...
uv run python scripts/train.py --config-name wind/train_x2 models=rcan
```

### 分布式训练

当前仅支持 PyTorch DDP，使用 `torchrun` 启动。

```bash
torchrun --nproc_per_node=4 scripts/train.py
```

## 评估

这里的评估指“配对评估”：输入为成对的 `LR/HR` 数据，计算指标。

### 最小评估 YAML

```yaml
defaults:
  - models: edsr
  - _self_

upscale: 2

data_norm:
  mean: [4.565732]
  std: [3.1361]

dataset:
  name: wind
  eval_lr_root: ./data/wind/test/x2/lr
  eval_hr_root: ./data/wind/test/x2/hr
  max_sample: false # false=全量；正整数=只取前 N 个评估样本

evaluate:
  checkpoint: ./checkpoints/.../best.pth
  batch_size: 1
  save_results: false
  save_dir: ./results
  mask: null
  num_workers: 4
```

### 启动命令

```bash
uv run python scripts/evaluate.py evaluate.checkpoint=./checkpoints/.../best.pth
```

常用 override 示例：

```bash
uv run python scripts/evaluate.py models=camixer evaluate.checkpoint=./checkpoints/.../best.pth
uv run python scripts/evaluate.py dataset.eval_lr_root=./data/... dataset.eval_hr_root=./data/...
uv run python scripts/evaluate.py --config-name wind/evaluate_x2 evaluate.checkpoint=./checkpoints/.../best.pth evaluate.batch_size=4
```

## 推理

这里的推理指“离线推理”：只输入 `LR`，不要求提供 `HR`。

### 最小推理 YAML

```yaml
defaults:
  - models: edsr
  - _self_

upscale: 2

data_norm:
  mean: [4.565732]
  std: [3.1361]

dataset:
  name: wind
  infer_lr_root: ./data/wind/infer/x2/lr

infer:
  checkpoint: ./checkpoints/.../best.pth
  batch_size: 1
  save_results: true
  save_dir: ./results
  mask: null
  num_workers: 4
```

### 启动命令

```bash
uv run python scripts/infer.py infer.checkpoint=./checkpoints/.../best.pth
```

常用 override 示例：

```bash
uv run python scripts/infer.py models=edsr infer.checkpoint=./checkpoints/.../best.pth
uv run python scripts/infer.py dataset.infer_lr_root=./data/...
uv run python scripts/infer.py --config-name wind/infer_x2 infer.checkpoint=./checkpoints/.../best.pth infer.batch_size=4
```

## 断点续训

恢复训练时：

- `train.epochs` 表示总轮数，不是追加轮数
- 例如 checkpoint 已训练到第 40 轮，若当前配置是 `epochs: 200`，则会从第 41 轮继续训练到第 200 轮

### 配置示例

```yaml
resume:
  checkpoint: ./checkpoints/.../last.pth
  load_optimizer: true
  load_scheduler: true
  load_callbacks: true
  load_rng_state: true
```

### 启动命令

```bash
uv run python scripts/train.py resume.checkpoint=./checkpoints/.../last.pth
```

恢复时会同步加载：

- 模型参数
- `epoch`
- `global_step`
- 配置快照

在此基础上，还会按 `resume` 开关选择性恢复：

- 优化器状态：`load_optimizer`
- 调度器状态：`load_scheduler`
- callback 状态：`load_callbacks`
- 随机数状态：`load_rng_state`

其中最佳指标相关状态现在保存在 `CheckpointCallback` 内部，而不是 `Engine` 顶层字段。

程序还会检查 checkpoint 保存时的关键配置签名是否与当前任务一致，当前比对包括：

- `models.name`
- `models.in_dim`
- `models.params`
- `upscale`

## 数据格式与目录约定

当前数据以 `.npy` 为基本格式。

### 训练

训练必须显式提供：

- `train_lr_root`
- `train_hr_root`
- `val_lr_root`
- `val_hr_root`

### 评估

评估必须显式提供：

- `eval_lr_root`
- `eval_hr_root`

### 推理

推理只需要显式提供：

- `infer_lr_root`

### 目录示例

```text
data/
└── wind/
    ├── train/
    │   ├── x2/
    │   │   ├── lr/*.npy
    │   │   └── hr/*.npy
    │   └── x4/
    │       ├── lr/*.npy
    │       └── hr/*.npy
    ├── test/
    │   ├── x2/
    │   │   ├── lr/*.npy
    │   │   └── hr/*.npy
    │   └── x4/
    │       ├── lr/*.npy
    │       └── hr/*.npy
    └── infer/
        ├── x2/lr/*.npy
        └── x4/lr/*.npy
```

要求：

- 训练和评估阶段，`LR/HR` 文件名必须一一对应
- 程序只负责读取、裁 patch、标准化与反标准化后的指标计算
- `LR/HR`、通道组织方式以及 `mean/std` 由实验人员自行准备
- `wind`、`mwp`、`swh` 当前按 1 通道处理
- `mwd` 当前按 2 通道处理
- 单通道数据可以直接存为 `HxW`
- 多通道数据可以直接存为 `CxHxW`，并提供匹配通道数的 `mean/std`
- 使用 `scripts/preprocess_dataset.py` 预处理后，导出的 HR/LR 会统一保存为 `C×H×W`

## 输出目录说明

### `paths.checkpoint_dir`

用于保存权重文件，例如：

- `epoch_010.pth`
- `best.pth`
- `last.pth`

含义：

- `epoch_xxx.pth`：周期性保存
- `best.pth`：当前最佳指标对应的 checkpoint
- `last.pth`：最近一次训练状态

### `paths.experiment_dir`

用于保存运行目录、日志、Hydra 输出和配置快照。

评估与推理在启用结果保存时，也会在 `save_dir/<Model>/<dataset>/x{upscale}/` 下按样本写出 `.npy` 结果。

## 模型与损失

### 当前内置模型

- `edsr`
- `rcan`
- `rdn`
- `swinir`
- `mysr`
- `mysrab1`
- `mysrab2`
- `atd`
- `camixer`

### 当前内置损失

- `l1`
- `mae`
- `l2`
- `mse`

## 新增模型如何接入

当前模型系统基于注册机制，不再依赖 `if-elif` 分发。

新增模型的标准步骤：

1. 在 `src/models/` 新建模型文件
2. 用 `@register_model(...)` 注册工厂函数
3. 返回标准模型结果字典
4. 在 `configs/models/` 下新增对应 YAML
5. 用训练 / 评估 / 推理三条主流程验证

最核心的是两点：

- 模型需要能通过注册表被创建
- 模型前向输出需要能被统一训练逻辑消费

当前训练引擎支持这些输出形式：

- 直接返回预测张量 `pred`
- 返回 `(pred, aux_loss)`
- 返回 `(pred, {"aux_losses": {...}})`

这意味着新模型不必手工接入单独的训练分支，只要遵守统一输出约定，就能复用现有训练、评估、推理流程。

## 运行前校验与常见错误

在真正创建 dataloader 前，程序会做运行前校验。

### 训练校验

- `train_lr_root` / `train_hr_root` / `val_lr_root` / `val_hr_root` 必须存在
- 必须是目录
- 目录下必须包含 `.npy`
- `LR/HR` 文件名必须能正确配对

### 评估校验

- `eval_lr_root` / `eval_hr_root` 必须存在
- 必须是目录
- 必须有可配对样本
- `evaluate.mask` 若填写，必须是 `.npy`

### 推理校验

- `infer_lr_root` 必须存在
- 必须是目录
- 至少包含一个 `.npy`
- `infer.mask` 若填写，必须是 `.npy`

### 其他配置校验

- `LR/HR` 路径不能相同
- `train.checkpoint.every` 不能大于 `train.epochs`
- `resume.checkpoint` / `evaluate.checkpoint` / `infer.checkpoint` 必须是 `.pth` 或 `.pt`

如果启动即报错，优先检查：

1. 路径是否写错
2. `LR/HR` 是否同名配对
3. 模型名与配置文件是否一致
4. checkpoint 是否和当前模型 / 参数配置匹配

## 测试

完整测试：

```bash
uv run pytest tests -q
```

核心回归测试：

```bash
uv run pytest tests/test_config.py tests/test_engine.py tests/test_evaluator.py tests/test_metrics.py tests/test_run_queue.py -q
```

## 建议的阅读顺序

如果你是第一次接手这个项目，建议按下面顺序看：

1. `configs/train.yaml`、`configs/evaluate.yaml`、`configs/infer.yaml`
2. `configs/wind/train_x2.yaml` 这类按数据集拆开的任务配置
3. `configs/models/*.yaml`
4. `scripts/train.py`、`scripts/evaluate.py`、`scripts/infer.py`
5. `src/app/` 和 `src/core/`
6. `src/models/` 与 `src/losses/`
