"""
测试优化器和调度器
"""

import torch
import torch.nn as nn

from src.optim.optimizer import get_optimizer, list_optimizers
from src.optim.scheduler import SchedulerController, get_scheduler, list_schedulers


def test_list_optimizers():
    optimizers = list_optimizers()
    assert isinstance(optimizers, list)
    assert "adam" in optimizers
    assert "adamw" in optimizers


def test_list_schedulers():
    schedulers = list_schedulers()
    assert isinstance(schedulers, list)
    assert "cosine" in schedulers
    assert "step" in schedulers


def test_get_optimizer():
    model = nn.Linear(10, 10)
    optimizer = get_optimizer("adamw", model, lr=1e-4)

    assert isinstance(optimizer, torch.optim.AdamW)


def test_get_scheduler():
    model = nn.Linear(10, 10)
    optimizer = get_optimizer("adam", model, lr=1e-4)
    scheduler = get_scheduler("cosine", optimizer, T_max=100)

    assert isinstance(scheduler, SchedulerController)
    assert isinstance(scheduler.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert scheduler.step_mode == "epoch"
