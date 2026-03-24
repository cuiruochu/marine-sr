import torch
import torch.nn as nn

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
