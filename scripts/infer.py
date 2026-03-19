"""
离线推理入口

只要求输入 LR 数据。
"""

import hydra
from omegaconf import DictConfig

from src.app.eval import run_evaluation


@hydra.main(config_path="../configs", config_name="infer", version_base=None)
def main(cfg: DictConfig):
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
