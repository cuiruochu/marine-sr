"""离线推理入口。"""

import hydra
from omegaconf import DictConfig

from src.app.eval import run_inference


@hydra.main(config_path="../configs", config_name="infer", version_base=None)
def main(cfg: DictConfig):
    run_inference(cfg)


if __name__ == "__main__":
    main()
