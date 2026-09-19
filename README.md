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


## 1. 配置环境

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


## 2. 项目结构

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

## 3. 配置设计

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

## 4. 训练

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
uv run python scripts/train.py \
  --config-name wind/train_x2 \
  models=mysr
```

## 5. 评估

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

### 启动命令（以 mwp 为例）

```bash
uv run python scripts/evaluate.py \
  --config-name mwp/evaluate_x4 \
  models=mysr \
  +models.in_dim=1 \
  evaluate.checkpoint=./checkpoints/mysr/mwp/x4/last.pth \
  dataset.eval_lr_root=../dataset/test/mwp/x4/lr \
  dataset.eval_hr_root=../dataset/test/mwp/x4/hr \
  evaluate.mask=../dataset/test/mwp/x4/mask.npy
```

以 mwd 为例（注意 `in_dim=2`，且 mwd 的 `data_norm` 有 2 个通道）：

```bash
uv run python scripts/evaluate.py \
  --config-name mwd/evaluate_x4 \
  models=mysr \
  +models.in_dim=2 \
  evaluate.checkpoint=./checkpoints/mysr/mwd/x4/last.pth \
  dataset.eval_lr_root=../dataset/test/mwd/x4/lr \
  dataset.eval_hr_root=../dataset/test/mwd/x4/hr \
  evaluate.mask=../dataset/test/mwd/x4/mask.npy
```

## 6. 推理

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
uv run python scripts/infer.py \
  --config-name mwp/infer_x4 \
  models=mysr \
  +models.in_dim=1 \
  infer.checkpoint=./checkpoints/mysr/mwp/x4/last.pth \
  dataset.infer_lr_root=../dataset/test/mwp/x4/lr \
  infer.mask=../dataset/test/mwp/x4/mask.npy
```


## 7. 断点续训

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

## 8. 串行实验队列

如果你需要“一个实验结束立刻跑下一个，失败也继续下一个”，可以使用队列执行器：

```bash
uv run python scripts/run_queue.py --queue jobs/queue.txt
```

当前 `jobs/queue.txt` 是训练队列模板，覆盖：

- 全部内置模型
- `wind` / `mwd` / `mwp` / `swh` 四个数据集

只检查任务而不真正执行：

```bash
uv run python scripts/run_queue.py --queue jobs/queue.txt --dry-run
```

## 9. 数据格式与目录约定

当前数据以 `.npy` 为基本格式。

### Checkpoint 默认保存路径

Checkpoint 会按 `<checkpoint_dir>/<模型名>/<数据集名>/x<倍数>/` 自动组织。例如 `mysr` 训练 `mwp` 数据集 `x4` 时，保存路径为：

```text
checkpoints/mysr/mwp/x4/
├── best.pth      # 验证指标最优的权重
├── last.pth      # 最近一次训练结束时的权重
└── epoch_XXX.pth # 按 train.checkpoint.every 周期保存的权重
```

### 数据存放目录示例

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
- 数据可以直接存为 `CxHxW`，并提供匹配通道数的 `mean/std`

### 数据预处理

使用 NetCDF 构建数据集时，可以通过统一流水线直接生成最终的 `HR/LR` 配对数据，不保存中间文件：

```bash
uv run python scripts/build_dataset.py --oper-nc ../dataset-2022/data_stream-oper_stepType-instant.nc --wave-nc ../dataset-2022/data_stream-wave_stepType-instant.nc --output-dir ../dataset-2022/test --scales 2 4
```

`--scales` 支持传入一个或多个放大倍数，例如 `--scales 2`、`--scales 4` 或 `--scales 2 4`。传入多个尺度时，流水线会按最大尺度统一裁剪 HR 区域，再基于同一个 HR 生成所有尺度的 LR，保证不同倍率使用同一片空间范围。

流水线输入要求：

- `--oper-nc` 必须包含 `u10` 和 `v10`
- `--wave-nc` 必须包含 `mwd`、`mwp` 和 `swh`
- `--wave-nc` 中所有变量必须来自同一片空间范围，因此可以共用同一个有效区域 mask

流水线会生成：

```text
../dataset-2022/test/
├── wind/x2/hr/*.npy
├── wind/x2/lr/*.npy
├── wind/x4/hr/*.npy
├── wind/x4/lr/*.npy
├── mwd/x2/hr/*.npy
├── mwd/x2/lr/*.npy
├── mwd/x2/mask.npy
├── mwd/x4/mask.npy
├── mwp/x2/...
├── swh/x2/...
└── ...
```

处理规则：

- `wind` 由 `u10` 和 `v10` 通过 `sqrt(u10^2 + v10^2)` 合成
- `mwd` 先按角度制编码为 `cos/sin` 双通道，再生成 `HR/LR`
- `mwd`、`mwp`、`swh` 会保存 `H×W` 的 `bool` 类型 `mask.npy`
- mask 取第一个时间帧生成，`True` 表示有效区域，`False` 表示 `NaN` 或无效区域
- 原始数据中的 `NaN` 会填成 `0`
- 所有数组保存为 `float32`
- 测试集不单独统计 `mean/std`，归一化统计量应来自训练集

## 10. 模型与损失

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

### 新增模型如何接入


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

## 11. 测试

完整测试：

```bash
uv run pytest tests -q
```

核心回归测试：

```bash
uv run pytest tests/test_config.py tests/test_engine.py tests/test_evaluator.py tests/test_metrics.py tests/test_run_queue.py -q
```
