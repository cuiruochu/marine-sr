"""评估指标回调。"""

import logging

from src.core.callbacks import Callback

logger = logging.getLogger(__name__)


class MetricsCallback(Callback):
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def on_eval_end(self, evaluator, metrics: dict[str, float]) -> None:
        if not self.verbose or not metrics:
            return

        logger.info("=" * 50)
        logger.info("Evaluation Results")
        logger.info("=" * 50)
        logger.info(f"  PSNR:    {metrics['psnr']:.4f} dB")
        logger.info(f"  SSIM:    {metrics['ssim']:.4f}")
        logger.info(f"  MAE:     {metrics['mae']:.4f}")
        logger.info(f"  Max MAE: {metrics['max_mae']:.4f}")
        logger.info("=" * 50)
