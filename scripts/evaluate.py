"""配对评估入口。"""

import hydra
from omegaconf import DictConfig

from src.app.eval import run_evaluation


@hydra.main(config_path="../configs", config_name="evaluate", version_base=None)
def main(cfg: DictConfig):
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
