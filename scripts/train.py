"""
统一训练入口

训练编排逻辑已迁移到 src.app.train，脚本层只保留 Hydra 入口。
"""

import hydra
from omegaconf import DictConfig

from src.app.train import run_training


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    run_training(cfg)


if __name__ == "__main__":
    main()
