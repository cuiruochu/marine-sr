"""应用运行时公共工具。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.utils.logging import configure_logging


def build_config_snapshot(raw_cfg: DictConfig | Any, cfg: Any) -> dict[str, Any]:
    if isinstance(raw_cfg, DictConfig):
        return OmegaConf.to_container(raw_cfg, resolve=True)
    return asdict(cfg)


def hydra_output_file(filename: str) -> Path | None:
    try:
        from hydra.core.hydra_config import HydraConfig

        output_dir = HydraConfig.get().runtime.output_dir
    except Exception:
        return None
    return Path(output_dir) / filename


def hydra_run_name() -> str | None:
    output_path = hydra_output_file(".")
    if output_path is None:
        return None
    return output_path.parent.name


def init_task_logger(filename: str) -> None:
    configure_logging(file_path=hydra_output_file(filename))
