# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Marine Parameter Super-Resolution (SR) deep learning project. Upscales low-resolution marine meteorological data to high-resolution using various SR architectures.

## Common Commands

### Training
```bash
cd api && python train.py
```
Configure training in `config/single_train_config.py` (single parameter) or `config/unified_train_config.py` (unified model).

### Evaluation
```bash
cd api && python eval.py
```
Configure evaluation in `config/test_config.py`.

### Learning Rate Search
```bash
cd api && python find_lr.py
```

## Architecture

```
api/           # Entry points (train.py, eval.py)
config/        # Dataclass configurations (single_train_config.py, unified_train_config.py, test_config.py)
dataset/       # Data loaders and preprocessing (dataset.py, utils.py)
models/        # SR model implementations
trainers/      # Training logic (common_trainer.py, unified_trainer.py)
tester/        # Evaluation logic (common_tester.py, unified_tester.py)
```

## Training Modes

- **Single**: Train on one marine parameter. Config: `single_train_config.py`
- **Unified**: Train on all 4 parameters with shared encoder/decoder. Config: `unified_train_config.py`

## Marine Parameters

| Parameter | Channels | Description |
|-----------|----------|-------------|
| wind      | 1        | Wind speed  |
| mwd       | 2        | Mean wave direction (cos/sin encoding) |
| mwp       | 1        | Mean wave period |
| swh       | 1        | Significant wave height |

## Available Models

`Bicubic`, `EDSR`, `RCAN`, `RDN`, `SwinIR`, `ATD`, `CAMixer`, `MySR`, `MySRAb1`, `MySRAb2`

## Configuration Flow

1. `config/__init__.py::get_config(mode, unified)` returns appropriate config
2. Models created via `models/__init__.py::create_model(config)`
3. Trainers selected via `trainers/__init__.py::get_trainer(...)`
4. Testers selected via `tester/__init__.py::get_tester(...)`

## Key Implementation Details

- MWD parameter uses 2-channel encoding (cos/sin) defined in `dataset/utils.py::encode_mwd`
- Normalization per parameter defined in `NormalizeConfig` dataclass
- Unified model uses separate encoders/decoders for mwd (2ch) vs others (1ch)
- Metrics: PSNR, SSIM, MAE calculated in `tester/utils.py`
- Checkpoints saved to `../checkpoints/{model_name}/{marine_param}/x{scale}/`