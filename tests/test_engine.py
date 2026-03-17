"""
Engine 单元测试
"""

import pytest
import torch
import torch.nn as nn
import tempfile
from pathlib import Path

from src.core import Engine, Callback


class DummyModel(nn.Module):
    """测试用简单模型"""

    def __init__(self, in_dim=1, out_dim=1, upscale=1):
        super().__init__()
        self.upscale = upscale
        self.conv = nn.Conv2d(in_dim, out_dim, 3, padding=1)
        # 上采样层
        if upscale > 1:
            self.upsample = nn.Upsample(scale_factor=upscale, mode='bilinear', align_corners=False)
        else:
            self.upsample = None

    def forward(self, x):
        x = self.conv(x)
        if self.upsample:
            x = self.upsample(x)
        return x


class DummyDataset(torch.utils.data.Dataset):
    """测试用数据集"""

    def __init__(self, size=10, in_dim=1, upscale=2):
        self.size = size
        self.in_dim = in_dim
        self.upscale = upscale

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        lr = torch.randn(self.in_dim, 16, 16)
        # HR 尺寸是 LR 的 upscale 倍
        hr = torch.randn(self.in_dim, 16 * self.upscale, 16 * self.upscale)
        return lr, hr


class TestEngine:
    """Engine 测试"""

    def test_engine_init(self):
        """测试 Engine 初始化"""
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters())
        engine = Engine(model=model, optimizer=optimizer, loss_fn=nn.L1Loss())

        assert engine.model is model
        assert engine.optimizer is optimizer
        assert engine.current_epoch == 0
        assert engine.global_step == 0

    def test_train_step(self):
        """测试训练步骤"""
        model = DummyModel(upscale=2)
        optimizer = torch.optim.Adam(model.parameters())
        engine = Engine(model=model, optimizer=optimizer, loss_fn=nn.L1Loss())

        batch = (torch.randn(1, 1, 16, 16), torch.randn(1, 1, 32, 32))
        logs = engine.train_step(batch)

        assert "loss" in logs
        assert isinstance(logs["loss"], float)

    def test_evaluate(self):
        """测试评估"""
        model = DummyModel(upscale=2)
        optimizer = torch.optim.Adam(model.parameters())
        engine = Engine(model=model, optimizer=optimizer, loss_fn=nn.L1Loss())

        val_loader = torch.utils.data.DataLoader(
            DummyDataset(size=5, upscale=2), batch_size=2
        )
        metrics = engine.evaluate(val_loader, mean=[0], std=[1])

        assert "val_psnr" in metrics
        assert "val_ssim" in metrics
        assert "val_mae" in metrics

    def test_fit(self):
        """测试完整训练"""
        model = DummyModel(upscale=2)
        optimizer = torch.optim.Adam(model.parameters())
        engine = Engine(model=model, optimizer=optimizer, loss_fn=nn.L1Loss())

        train_loader = torch.utils.data.DataLoader(
            DummyDataset(size=4, upscale=2), batch_size=2
        )
        val_loader = torch.utils.data.DataLoader(
            DummyDataset(size=2, upscale=2), batch_size=2
        )

        history = engine.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=2,
            mean=[0],
            std=[1],
        )

        assert len(history["train_loss"]) == 2
        assert len(history["val_psnr"]) == 2
        assert engine.current_epoch == 2

    def test_save_load_checkpoint(self):
        """测试检查点保存和加载"""
        model = DummyModel(upscale=2)
        optimizer = torch.optim.Adam(model.parameters())
        engine = Engine(model=model, optimizer=optimizer, loss_fn=nn.L1Loss())

        # 模拟训练
        engine.current_epoch = 5
        engine.global_step = 100

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pth"
            engine.save_checkpoint(str(path), epoch=5)

            # 创建新 engine 并加载
            new_model = DummyModel(upscale=2)
            new_optimizer = torch.optim.Adam(new_model.parameters())
            new_engine = Engine(model=new_model, optimizer=new_optimizer, loss_fn=nn.L1Loss())

            loaded_epoch = new_engine.load_checkpoint(str(path))

            assert loaded_epoch == 5
            assert new_engine.current_epoch == 5
            assert new_engine.global_step == 100

    def test_count_parameters(self):
        """测试参数统计"""
        model = DummyModel(in_dim=1, out_dim=1, upscale=2)
        optimizer = torch.optim.Adam(model.parameters())
        engine = Engine(model=model, optimizer=optimizer, loss_fn=nn.L1Loss())

        total = engine.count_parameters(trainable_only=True)

        assert total > 0
        assert total == sum(p.numel() for p in model.parameters() if p.requires_grad)

    def test_lr_property(self):
        """测试学习率属性"""
        model = DummyModel(upscale=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        engine = Engine(model=model, optimizer=optimizer, loss_fn=nn.L1Loss())

        assert engine.lr == 1e-4


class TestCallback:
    """Callback 测试"""

    def test_callback_hooks(self):
        """测试回调钩子"""

        class CountingCallback(Callback):
            def __init__(self):
                self.call_count = 0

            def on_train_begin(self, engine):
                self.call_count += 1

            def on_train_end(self, engine):
                self.call_count += 1

            def on_epoch_begin(self, engine, epoch):
                self.call_count += 1

            def on_epoch_end(self, engine, epoch, logs):
                self.call_count += 1

        model = DummyModel(upscale=2)
        optimizer = torch.optim.Adam(model.parameters())
        callback = CountingCallback()
        engine = Engine(
            model=model,
            optimizer=optimizer,
            loss_fn=nn.L1Loss(),
            callbacks=[callback],
        )

        train_loader = torch.utils.data.DataLoader(DummyDataset(size=2, upscale=2), batch_size=1)
        val_loader = torch.utils.data.DataLoader(DummyDataset(size=1, upscale=2), batch_size=1)

        engine.fit(train_loader, val_loader, epochs=2, mean=[0], std=[1])

        # on_train_begin + on_train_end + 2 * (on_epoch_begin + on_epoch_end)
        assert callback.call_count == 6