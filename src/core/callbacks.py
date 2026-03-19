"""
回调机制

提供训练和评估过程中的生命周期钩子，实现逻辑解耦。
"""

from typing import Dict, Any, List, Optional
from abc import ABC


class Callback(ABC):
    """
    回调基类

    训练生命周期钩子：
    - on_train_begin/end: 训练开始/结束
    - on_epoch_begin/end: 每个 epoch 开始/结束
    - on_batch_begin/end: 每个 batch 开始/结束

    评估生命周期钩子：
    - on_eval_begin/end: 评估开始/结束
    - on_batch_begin/end: 每个 batch 开始/结束（评估时也触发）

    用法：
        class MyCallback(Callback):
            def on_epoch_end(self, engine, epoch, logs):
                logger.info(f"Epoch {epoch}: {logs}")
    """

    def state_dict(self) -> Dict[str, Any]:
        """返回需要写入 checkpoint 的回调状态。"""
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """从 checkpoint 恢复回调状态。"""
        return None

    # === 训练回调 ===

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

    def on_batch_begin(self, engine: Any, batch: int) -> None:
        """每个 batch 开始时调用"""
        pass

    def on_batch_end(self, engine: Any, batch: int, logs: Dict[str, Any] = None, **kwargs) -> None:
        """每个 batch 结束时调用

        Args:
            engine: 引擎实例（Engine 或 Evaluator）
            batch: 当前 batch 索引
            logs: 指标字典（训练时包含 loss，评估时可能为 None）
            **kwargs: 额外参数（评估时包含 sr, hr, filename, metrics 等）
        """
        pass

    # === 评估回调 ===

    def on_eval_begin(self, evaluator: "Evaluator") -> None:
        """评估开始时调用"""
        pass

    def on_eval_end(self, evaluator: "Evaluator", metrics: Dict[str, float]) -> None:
        """评估结束时调用

        Args:
            evaluator: 评估引擎实例
            metrics: 指标字典，包含 psnr, ssim, mae, max_mae
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

    def on_batch_begin(self, engine: Any, batch: int) -> None:
        for cb in self.callbacks:
            cb.on_batch_begin(engine, batch)

    def on_batch_end(self, engine: Any, batch: int, logs: Dict[str, Any] = None, **kwargs) -> None:
        for cb in self.callbacks:
            cb.on_batch_end(engine, batch, logs=logs, **kwargs)

    def on_eval_begin(self, evaluator: "Evaluator") -> None:
        for cb in self.callbacks:
            cb.on_eval_begin(evaluator)

    def on_eval_end(self, evaluator: "Evaluator", metrics: Dict[str, float]) -> None:
        for cb in self.callbacks:
            cb.on_eval_end(evaluator, metrics)

    def append(self, callback: Callback) -> None:
        """添加回调"""
        self.callbacks.append(callback)

    def state_dict(self) -> Dict[str, Dict[str, Any]]:
        """收集所有回调状态。"""
        state = {}
        for idx, callback in enumerate(self.callbacks):
            key = f"{callback.__class__.__module__}.{callback.__class__.__qualname__}:{idx}"
            callback_state = callback.state_dict()
            if callback_state:
                state[key] = callback_state
        return state

    def load_state_dict(self, state: Dict[str, Dict[str, Any]] | None) -> None:
        """恢复所有回调状态。"""
        if not state:
            return

        for idx, callback in enumerate(self.callbacks):
            key = f"{callback.__class__.__module__}.{callback.__class__.__qualname__}:{idx}"
            if key in state:
                callback.load_state_dict(state[key])
