"""统一训练入口。"""

from pathlib import Path
import sys

import hydra
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.train import run_training


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    run_training(cfg)


if __name__ == "__main__":
    main()
