"""日志回调"""

import logging

from src.core.callbacks import Callback

logger = logging.getLogger(__name__)


class LoggingCallback(Callback):
    """训练日志回调"""

    def on_epoch_end(self, engine, epoch: int, logs: dict) -> None:
        logger.info(
            f"Epoch {epoch} finished: "
            f"loss={logs.get('train_loss', 0):.4f}, "
            f"val_loss={logs.get('val_loss', 0):.4f}, "
            f"lr={logs.get('lr', 0):.2e}"
        )
