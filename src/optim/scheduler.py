"""学习率调度器模块。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from torch.optim import lr_scheduler


@dataclass(frozen=True)
class SchedulerSpec:
    factory: type | None
    step_mode: Literal["epoch", "metric"] = "epoch"
    monitor: str = "val_loss"


class SchedulerController:
    """统一封装 scheduler 的 step 契约。"""

    def __init__(self, scheduler, *, step_mode: Literal["epoch", "metric"], monitor: str = "val_loss"):
        self.scheduler = scheduler
        self.step_mode = step_mode
        self.monitor = monitor

    def step(self, metrics: dict[str, Any] | None = None) -> None:
        if self.step_mode == "metric":
            if metrics is None or self.monitor not in metrics:
                raise ValueError(f"调度器需要监控指标 '{self.monitor}'，但当前日志中不存在该字段")
            self.scheduler.step(metrics[self.monitor])
            return

        self.scheduler.step()

    def state_dict(self) -> dict[str, Any]:
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.scheduler.load_state_dict(state_dict)


SCHEDULER_REGISTRY = {
    "step": SchedulerSpec(lr_scheduler.StepLR),
    "multistep": SchedulerSpec(lr_scheduler.MultiStepLR),
    "cosine": SchedulerSpec(lr_scheduler.CosineAnnealingLR),
    "plateau": SchedulerSpec(lr_scheduler.ReduceLROnPlateau, step_mode="metric", monitor="val_loss"),
    "none": SchedulerSpec(None),
}


def _normalize_scheduler_spec(entry) -> SchedulerSpec:
    if isinstance(entry, SchedulerSpec):
        return entry
    if entry is None:
        return SchedulerSpec(None)
    return SchedulerSpec(entry)


def register_scheduler(name: str, *, step_mode: Literal["epoch", "metric"] = "epoch", monitor: str = "val_loss"):
    """注册调度器的装饰器。"""

    def decorator(cls):
        SCHEDULER_REGISTRY[name] = SchedulerSpec(cls, step_mode=step_mode, monitor=monitor)
        return cls

    return decorator


def get_scheduler(name: str, optimizer, **kwargs):
    """创建带统一 step 接口的调度器实例。"""
    if name.lower() == "none" or not name:
        return None

    if name not in SCHEDULER_REGISTRY:
        raise ValueError(f"调度器 '{name}' 未注册。可用: {list(SCHEDULER_REGISTRY.keys())}")

    spec = _normalize_scheduler_spec(SCHEDULER_REGISTRY[name])
    if spec.factory is None:
        return None

    scheduler = spec.factory(optimizer, **kwargs)
    return SchedulerController(scheduler, step_mode=spec.step_mode, monitor=spec.monitor)


def list_schedulers() -> list:
    return list(SCHEDULER_REGISTRY.keys())
