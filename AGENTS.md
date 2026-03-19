# Repository Guidelines

## Project Structure & Module Organization
Core code lives in `src/`, organized by responsibility: `models/`, `datasets/`, `core/`, `callbacks/`, `losses/`, `optim/`, `trainers/`, `testers/`, and shared utilities in `utils/`. Runtime entrypoints are in `scripts/` (`train.py`, `eval.py`). Hydra configuration is split under `configs/` by concern, such as `configs/model/`, `configs/dataset/`, `configs/train/`, and `configs/eval/`. Tests are in `tests/` and mirror the code layout with files like `test_models.py` and `test_engine.py`. Store generated checkpoints under `checkpoints/`; do not commit large artifacts unless explicitly required.

## Build, Test, and Development Commands
Use `uv` for environment and dependency management.

- `uv sync` installs the core runtime dependencies.
- `uv sync --extra all` installs optional model dependencies such as ATD and CAMixer.
- `uv run python scripts/train.py` starts single-GPU training with default Hydra config.
- `torchrun --nproc_per_node=4 scripts/train.py` runs distributed training with the same entrypoint.
- `uv run python scripts/eval.py checkpoint=./checkpoints/.../best.pth` evaluates a trained checkpoint.
- `uv run pytest tests/ -v` runs the full test suite.
- `uv run pytest tests/test_models.py -v` targets a single module during development.
- `uv run pytest tests/ -v --cov=src` checks coverage on touched modules.
- `uv run ruff check .` runs linting.

## Coding Style & Naming Conventions
Target Python 3.10+ and keep code compatible with the package settings in `pyproject.toml`. Use 4-space indentation, snake_case for functions and modules, PascalCase for classes, and clear Hydra config names matching the existing lowercase pattern (`edsr`, `mwd`, `default`). Ruff is configured with a 120-character line limit; keep imports ordered and avoid unused symbols.

## Testing Guidelines
Write tests with `pytest`. Name files `test_*.py`, test functions `test_*`, and place fixtures in `tests/conftest.py` when they are shared. New work should include or update tests for the affected module path in `src/`. Prefer focused unit tests first, then run `uv run pytest tests/ -v` before opening a PR.

## Commit & Pull Request Guidelines
Recent history follows a lightweight Conventional Commit style such as `feat:`, `refactor:`, `test:`, and `docs:`. Keep commit subjects short and action-oriented. PRs should describe the changed module, the config or dataset impact, and the verification steps you ran. Include command output or metrics when training or evaluation behavior changes.
