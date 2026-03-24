"""
回调机制

提供训练和评估过程中的生命周期钩子，实现逻辑解耦。
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .engine import Engine


class Callback(ABC):
    """回调基类"""

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        return None

    def on_train_begin(self, engine: Engine) -> None:
        pass

    def on_train_end(self, engine: Engine) -> None:
        pass

    def on_epoch_begin(self, engine: Engine, epoch: int) -> None:
        pass

    def on_epoch_end(self, engine: Engine, epoch: int, logs: Dict[str, Any]) -> None:
        pass

    def on_batch_begin(self, engine: Any, batch: int) -> None:
        pass

    def on_batch_end(self, engine: Any, batch: int, logs: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        pass

    def on_eval_begin(self, evaluator: Any) -> None:
        pass

    def on_eval_end(self, evaluator: Any, metrics: Dict[str, float]) -> None:
        pass


class CallbackList:
    """回调列表，批量调用多个回调"""

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []

    def on_train_begin(self, engine: Engine) -> None:
        for cb in self.callbacks:
            cb.on_train_begin(engine)

    def on_train_end(self, engine: Engine) -> None:
        for cb in self.callbacks:
            cb.on_train_end(engine)

    def on_epoch_begin(self, engine: Engine, epoch: int) -> None:
        for cb in self.callbacks:
            cb.on_epoch_begin(engine, epoch)

    def on_epoch_end(self, engine: Engine, epoch: int, logs: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(engine, epoch, logs)

    def on_batch_begin(self, engine: Any, batch: int) -> None:
        for cb in self.callbacks:
            cb.on_batch_begin(engine, batch)

    def on_batch_end(self, engine: Any, batch: int, logs: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        for cb in self.callbacks:
            cb.on_batch_end(engine, batch, logs=logs, **kwargs)

    def on_eval_begin(self, evaluator: Any) -> None:
        for cb in self.callbacks:
            cb.on_eval_begin(evaluator)

    def on_eval_end(self, evaluator: Any, metrics: Dict[str, float]) -> None:
        for cb in self.callbacks:
            cb.on_eval_end(evaluator, metrics)

    def append(self, callback: Callback) -> None:
        self.callbacks.append(callback)

    def state_dict(self) -> Dict[str, Dict[str, Any]]:
        state = {}
        for idx, callback in enumerate(self.callbacks):
            key = f"{callback.__class__.__module__}.{callback.__class__.__qualname__}:{idx}"
            callback_state = callback.state_dict()
            if callback_state:
                state[key] = callback_state
        return state

    def load_state_dict(self, state: Optional[Dict[str, Dict[str, Any]]]) -> None:
        if not state:
            return

        for idx, callback in enumerate(self.callbacks):
            key = f"{callback.__class__.__module__}.{callback.__class__.__qualname__}:{idx}"
            if key in state:
                callback.load_state_dict(state[key])
