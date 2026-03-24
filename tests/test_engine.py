import torch
import torch.nn as nn
from pathlib import Path

from src.callbacks.checkpoint import CheckpointCallback
from src.core.engine import Engine
from src.optim.scheduler import get_scheduler


class IdentityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x + self.anchor * 0


class AuxLossModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.ones(1))

    def forward(self, x):
        pred = x + self.bias.view(1, 1, 1, 1)
        aux_loss = self.bias.square().mean()
        return pred, {"aux_losses": {"aux": aux_loss}}


def _checkpoint_path(test_name: str) -> Path:
    artifacts = Path(".test-artifacts") / "test_engine"
    artifacts.mkdir(parents=True, exist_ok=True)
    return artifacts / f"{test_name}.pt"


def test_engine_evaluate_returns_validation_loss_only():
    model = IdentityModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    engine = Engine(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.L1Loss(),
        callbacks=[],
        device="cpu",
    )

    hr = torch.randn(1, 2, 12, 12)
    val_loader = [(hr.clone(), hr.clone())]

    logs = engine.evaluate(val_loader=val_loader)

    assert logs == {"val_loss": 0.0}


def test_engine_train_and_eval_include_aux_loss():
    model = AuxLossModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    engine = Engine(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.L1Loss(),
        callbacks=[],
        device="cpu",
    )

    lr = torch.zeros(1, 1, 4, 4)
    hr = torch.ones(1, 1, 4, 4)

    train_logs = engine.train_step((lr.clone(), hr.clone()))
    eval_logs = engine.evaluate(val_loader=[(lr.clone(), hr.clone())])

    assert train_logs["base_loss"] == 0.0
    assert train_logs["aux_loss"] == 1.0
    assert train_logs["loss"] == 1.0
    assert eval_logs == {"val_loss": 1.0}


def test_engine_plateau_scheduler_steps_with_val_loss():
    model = IdentityModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = get_scheduler(
        "plateau",
        optimizer,
        mode="min",
        factor=0.5,
        patience=0,
        threshold=0.0,
        threshold_mode="abs",
        min_lr=0.0,
    )
    engine = Engine(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.L1Loss(),
        scheduler=scheduler,
        callbacks=[],
        device="cpu",
    )

    hr = torch.randn(1, 1, 4, 4)
    train_loader = [(hr.clone(), hr.clone())]
    val_loader = [(hr.clone(), hr.clone())]

    engine.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2,
    )

    assert engine.lr == 0.5


def test_load_checkpoint_skips_optimizer_scheduler_and_rng_when_disabled():
    path = _checkpoint_path("resume_flags_disabled")
    if path.exists():
        path.unlink()

    model = IdentityModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0, momentum=0.9)
    scheduler = get_scheduler(
        "plateau",
        optimizer,
        mode="min",
        factor=0.5,
        patience=0,
        threshold=0.0,
        threshold_mode="abs",
        min_lr=0.0,
    )
    checkpoint_callback = CheckpointCallback(save_dir=str(path.parent), monitor="val_loss")
    engine = Engine(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.L1Loss(),
        scheduler=scheduler,
        callbacks=[checkpoint_callback],
        device="cpu",
    )

    engine.current_epoch = 3
    engine.global_step = 7
    checkpoint_callback.best_value = 0.25
    checkpoint_callback.best_epoch = 2
    model.anchor.grad = torch.ones_like(model.anchor)
    optimizer.step()
    scheduler.step({"val_loss": 1.0})
    engine.save_checkpoint(str(path), epoch=3)

    optimizer.param_groups[0]["lr"] = 9.0
    optimizer.state.clear()
    scheduler.scheduler.best = 999.0
    scheduler.scheduler.num_bad_epochs = 5
    torch.manual_seed(999)
    rng_before = torch.get_rng_state().clone()

    loaded_epoch = engine.load_checkpoint(
        str(path),
        load_optimizer=False,
        load_scheduler=False,
        load_callbacks=False,
        load_rng_state=False,
    )

    assert loaded_epoch == 3
    assert optimizer.param_groups[0]["lr"] == 9.0
    assert optimizer.state == {}
    assert scheduler.scheduler.best == 999.0
    assert scheduler.scheduler.num_bad_epochs == 5
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert engine.global_step == 7
    assert checkpoint_callback.best_value == 0.25
    assert checkpoint_callback.best_epoch == 2

    path.unlink(missing_ok=True)


def test_load_checkpoint_restores_optimizer_scheduler_and_rng_when_enabled():
    path = _checkpoint_path("resume_flags_enabled")
    if path.exists():
        path.unlink()

    model = IdentityModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0, momentum=0.9)
    scheduler = get_scheduler(
        "plateau",
        optimizer,
        mode="min",
        factor=0.5,
        patience=0,
        threshold=0.0,
        threshold_mode="abs",
        min_lr=0.0,
    )
    checkpoint_callback = CheckpointCallback(save_dir=str(path.parent), monitor="val_loss")
    engine = Engine(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.L1Loss(),
        scheduler=scheduler,
        callbacks=[checkpoint_callback],
        device="cpu",
    )

    torch.manual_seed(123)
    expected_rng = torch.get_rng_state().clone()
    engine.current_epoch = 4
    engine.global_step = 9
    checkpoint_callback.best_value = 0.1
    checkpoint_callback.best_epoch = 4
    model.anchor.grad = torch.ones_like(model.anchor)
    optimizer.step()
    scheduler.step({"val_loss": 1.0})
    expected_lr = optimizer.param_groups[0]["lr"]
    expected_momentum = next(iter(optimizer.state.values()))["momentum_buffer"].clone()
    expected_scheduler_state = scheduler.state_dict()
    engine.save_checkpoint(str(path), epoch=4)

    optimizer.param_groups[0]["lr"] = 7.0
    optimizer.state.clear()
    scheduler.scheduler.best = 777.0
    scheduler.scheduler.num_bad_epochs = 9
    torch.manual_seed(999)

    loaded_epoch = engine.load_checkpoint(
        str(path),
        load_optimizer=True,
        load_scheduler=True,
        load_callbacks=True,
        load_rng_state=True,
    )

    assert loaded_epoch == 4
    assert optimizer.param_groups[0]["lr"] == expected_lr
    restored_state = next(iter(optimizer.state.values()))
    assert torch.equal(restored_state["momentum_buffer"], expected_momentum)
    assert scheduler.state_dict() == expected_scheduler_state
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert engine.global_step == 9
    assert checkpoint_callback.best_value == 0.1
    assert checkpoint_callback.best_epoch == 4

    path.unlink(missing_ok=True)


def test_load_checkpoint_skips_callbacks_when_disabled():
    path = _checkpoint_path("resume_callbacks_disabled")
    if path.exists():
        path.unlink()

    model = IdentityModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0, momentum=0.9)
    checkpoint_callback = CheckpointCallback(save_dir=str(path.parent), monitor="val_loss")
    engine = Engine(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.L1Loss(),
        callbacks=[checkpoint_callback],
        device="cpu",
    )

    engine.current_epoch = 2
    engine.global_step = 5
    checkpoint_callback.best_value = 0.2
    checkpoint_callback.best_epoch = 2
    engine.save_checkpoint(str(path), epoch=2)

    checkpoint_callback.best_value = 999.0
    checkpoint_callback.best_epoch = 999

    loaded_epoch = engine.load_checkpoint(
        str(path),
        load_callbacks=False,
    )

    assert loaded_epoch == 2
    assert checkpoint_callback.best_value == 999.0
    assert checkpoint_callback.best_epoch == 999

    path.unlink(missing_ok=True)
