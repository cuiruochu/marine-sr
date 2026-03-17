"""
Callbacks 单元测试
"""

import pytest
import torch
import torch.nn as nn
import tempfile
from pathlib import Path

from src.core import Engine
from src.callbacks import (
    CheckpointCallback,
    LoggingCallback,
    ProgressCallback,
)


class DummyModel(nn.Module):
    """测试用简单模型"""

    def __init__(self, upscale=2):
        super().__init__()
        self.upscale = upscale
        self.conv = nn.Conv2d(1, 1, 3, padding=1)
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

    def __init__(self, size=4, upscale=2):
        self.size = size
        self.upscale = upscale

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        lr = torch.randn(1, 16, 16)
        hr = torch.randn(1, 16 * self.upscale, 16 * self.upscale)
        return lr, hr


class TestCheckpointCallback:
    """CheckpointCallback 测试"""

    def test_save_every(self):
        """测试定期保存"""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = CheckpointCallback(
                save_dir=tmpdir,
                every=2,
                save_best=False,
            )

            model = DummyModel(upscale=2)
            optimizer = torch.optim.Adam(model.parameters())
            engine = Engine(
                model=model,
                optimizer=optimizer,
                loss_fn=nn.L1Loss(),
                callbacks=[callback],
            )

            train_loader = torch.utils.data.DataLoader(DummyDataset(upscale=2), batch_size=2)
            val_loader = torch.utils.data.DataLoader(DummyDataset(upscale=2), batch_size=2)

            engine.fit(train_loader, val_loader, epochs=4, mean=[0], std=[1])

            # 检查保存的文件
            saved_files = list(Path(tmpdir).glob("*.pth"))
            # epoch 2, 4 保存，加上 last.pth
            assert len(saved_files) >= 2

    def test_save_best(self):
        """测试保存最佳模型"""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = CheckpointCallback(
                save_dir=tmpdir,
                every=0,  # 不定期保存
                save_best=True,
                monitor="val_mae",
                mode="min",
            )

            model = DummyModel(upscale=2)
            optimizer = torch.optim.Adam(model.parameters())
            engine = Engine(
                model=model,
                optimizer=optimizer,
                loss_fn=nn.L1Loss(),
                callbacks=[callback],
            )

            train_loader = torch.utils.data.DataLoader(DummyDataset(upscale=2), batch_size=2)
            val_loader = torch.utils.data.DataLoader(DummyDataset(upscale=2), batch_size=2)

            engine.fit(train_loader, val_loader, epochs=2, mean=[0], std=[1])

            # 检查 best.pth 存在
            best_path = Path(tmpdir) / "best.pth"
            assert best_path.exists()

    def test_keep_last(self):
        """测试保留最近 N 个检查点"""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = CheckpointCallback(
                save_dir=tmpdir,
                every=1,
                save_best=False,
                keep_last=2,
            )

            model = DummyModel(upscale=2)
            optimizer = torch.optim.Adam(model.parameters())
            engine = Engine(
                model=model,
                optimizer=optimizer,
                loss_fn=nn.L1Loss(),
                callbacks=[callback],
            )

            train_loader = torch.utils.data.DataLoader(DummyDataset(upscale=2), batch_size=2)
            val_loader = torch.utils.data.DataLoader(DummyDataset(upscale=2), batch_size=2)

            engine.fit(train_loader, val_loader, epochs=4, mean=[0], std=[1])

            # 检查只保留最近 2 个
            epoch_files = list(Path(tmpdir).glob("epoch_*.pth"))
            # 加上 last.pth
            assert len(epoch_files) <= 2


class TestLoggingCallback:
    """LoggingCallback 测试"""

    def test_logging(self):
        """测试日志记录"""
        callback = LoggingCallback()

        model = DummyModel(upscale=2)
        optimizer = torch.optim.Adam(model.parameters())
        engine = Engine(
            model=model,
            optimizer=optimizer,
            loss_fn=nn.L1Loss(),
            callbacks=[callback],
        )

        train_loader = torch.utils.data.DataLoader(DummyDataset(upscale=2), batch_size=2)
        val_loader = torch.utils.data.DataLoader(DummyDataset(upscale=2), batch_size=2)

        # 不应该抛出异常
        engine.fit(train_loader, val_loader, epochs=1, mean=[0], std=[1])


class TestProgressCallback:
    """ProgressCallback 测试"""

    def test_progress(self):
        """测试进度条"""
        callback = ProgressCallback(leave=False)

        model = DummyModel(upscale=2)
        optimizer = torch.optim.Adam(model.parameters())
        engine = Engine(
            model=model,
            optimizer=optimizer,
            loss_fn=nn.L1Loss(),
            callbacks=[callback],
        )

        train_loader = torch.utils.data.DataLoader(DummyDataset(upscale=2), batch_size=2)
        val_loader = torch.utils.data.DataLoader(DummyDataset(upscale=2), batch_size=2)

        # 不应该抛出异常
        engine.fit(train_loader, val_loader, epochs=1, mean=[0], std=[1])