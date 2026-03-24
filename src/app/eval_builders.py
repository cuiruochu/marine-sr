"""评估/推理链路装配器。"""

from src.app.config_types import EvalTaskConfig
from src.callbacks.metrics_callback import MetricsCallback
from src.callbacks.save_results import SaveResultsCallback
from src.datamodules.eval_builders import build_test_loader as build_test_data_loader


def build_test_loader(cfg: EvalTaskConfig):
    return build_test_data_loader(
        cfg.test_loader_spec,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )


def build_eval_callbacks(cfg: EvalTaskConfig):
    callbacks = []

    if cfg.mode == "evaluation":
        callbacks.append(MetricsCallback(verbose=True))

    if cfg.save_results:
        callbacks.append(SaveResultsCallback(save_dir=str(cfg.results_root)))

    return callbacks
