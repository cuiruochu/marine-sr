"""
回调机制

提供训练过程中的生命周期钩子，实现训练逻辑解耦。
"""

from typing import Dict, Any, List, Optional
from abc import ABC


class Callback(ABC):
    """
    回调基类

    生命周期钩子：
    - on_train_begin/end: 训练开始/结束
    - on_epoch_begin/end: 每个 epoch 开始/结束
    - on_batch_begin/end: 每个 batch 开始/结束

    用法：
        class MyCallback(Callback):
            def on_epoch_end(self, engine, epoch, logs):
                print(f"Epoch {epoch}: {logs}")
    """

    def on_train_begin(self, engine: "Engine") -> None:
        """训练开始时调用"""
        pass

    def on_train_end(self, engine: "Engine") -> None:
        """训练结束时调用"""
        pass

    def on_epoch_begin(self, engine: "Engine", epoch: int) -> None:
        """每个 epoch 开始时调用"""
        pass

    def on_epoch_end(self, engine: "Engine", epoch: int, logs: Dict[str, Any]) -> None:
        """每个 epoch 结束时调用

        Args:
            engine: 训练引擎实例
            epoch: 当前 epoch
            logs: 指标字典，包含 val_psnr, val_ssim, val_mae, lr 等
        """
        pass

    def on_batch_begin(self, engine: "Engine", batch: int) -> None:
        """每个 batch 开始时调用"""
        pass

    def on_batch_end(self, engine: "Engine", batch: int, logs: Dict[str, Any]) -> None:
        """每个 batch 结束时调用

        Args:
            engine: 训练引擎实例
            batch: 当前 batch 索引
            logs: 指标字典，包含 loss 等
        """
        pass


class CallbackList:
    """
    回调列表，批量调用多个回调

    用法：
        callbacks = CallbackList([
            CheckpointCallback(...),
            LoggingCallback(),
        ])
        callbacks.on_epoch_end(engine, epoch, logs)
    """

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []

    def on_train_begin(self, engine: "Engine") -> None:
        for cb in self.callbacks:
            cb.on_train_begin(engine)

    def on_train_end(self, engine: "Engine") -> None:
        for cb in self.callbacks:
            cb.on_train_end(engine)

    def on_epoch_begin(self, engine: "Engine", epoch: int) -> None:
        for cb in self.callbacks:
            cb.on_epoch_begin(engine, epoch)

    def on_epoch_end(self, engine: "Engine", epoch: int, logs: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(engine, epoch, logs)

    def on_batch_begin(self, engine: "Engine", batch: int) -> None:
        for cb in self.callbacks:
            cb.on_batch_begin(engine, batch)

    def on_batch_end(self, engine: "Engine", batch: int, logs: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_batch_end(engine, batch, logs)

    def append(self, callback: Callback) -> None:
        """添加回调"""
        self.callbacks.append(callback)