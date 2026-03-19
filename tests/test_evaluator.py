"""
Evaluator 单元测试
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

from src.core import Evaluator, Callback, ModelOutput


class DummyDataset(Dataset):
    """简单测试数据集"""

    def __init__(self, num_samples=10, channels=1):
        self.num_samples = num_samples
        self.channels = channels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        lr = torch.randn(self.channels, 10, 10)
        hr = torch.randn(self.channels, 20, 20)
        return lr, hr, f"sample_{idx}.npy"


class DummyInferenceDataset(Dataset):
    """仅包含 LR 的推理数据集"""

    def __init__(self, num_samples=10, channels=1):
        self.num_samples = num_samples
        self.channels = channels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        lr = torch.randn(self.channels, 10, 10)
        return lr, f"sample_{idx}.npy"


class DummyModel(nn.Module):
    """简单测试模型"""

    def __init__(self, upscale=2, in_channels=1):
        super().__init__()
        self.upscale = upscale
        self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        # 简单上采样
        x = nn.functional.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        return self.conv(x)


class DummyTupleOutputModel(DummyModel):
    """返回 (pred, aux_loss) 的测试模型"""

    def forward(self, x):
        pred = super().forward(x)
        return pred, pred.abs().mean() * 0.1


class DummyStructuredOutputModel(DummyModel):
    """返回 ModelOutput 的测试模型"""

    def forward(self, x):
        pred = super().forward(x)
        return ModelOutput(pred=pred, aux_losses={"stability": pred.square().mean() * 0.01})


class TestEvaluator:
    """Evaluator 测试"""

    def test_evaluator_init(self):
        """测试初始化"""
        model = DummyModel()
        evaluator = Evaluator(model=model)

        assert evaluator.device in ["cuda", "cpu"]
        assert evaluator.model is not None
        assert len(evaluator.callbacks.callbacks) == 0

    def test_evaluator_with_callbacks(self):
        """测试带回调的初始化"""
        model = DummyModel()
        callbacks = [
            Callback(),
        ]
        evaluator = Evaluator(model=model, callbacks=callbacks)

        assert len(evaluator.callbacks.callbacks) == 1

    def test_evaluator_run(self):
        """测试评估运行"""
        model = DummyModel()
        evaluator = Evaluator(model=model)

        test_loader = DataLoader(DummyDataset(num_samples=5), batch_size=1)

        metrics = evaluator.run(
            test_loader=test_loader,
            mean=[0.0],
            std=[1.0],
            mask=None,
        )

        assert "psnr" in metrics
        assert "ssim" in metrics
        assert "mae" in metrics
        assert "max_mae" in metrics

    def test_evaluator_run_with_mask(self):
        """测试带掩码的评估"""
        model = DummyModel()
        evaluator = Evaluator(model=model)

        test_loader = DataLoader(DummyDataset(num_samples=5), batch_size=1)

        # 创建一个 bool 类型掩码 (1 表示有效区域)
        mask = np.ones((20, 20), dtype=bool)

        metrics = evaluator.run(
            test_loader=test_loader,
            mean=[0.0],
            std=[1.0],
            mask=mask,
        )

        assert "psnr" in metrics

    def test_evaluator_run_with_tuple_output(self):
        """评估阶段应忽略辅助损失，只使用 pred"""
        model = DummyTupleOutputModel()
        evaluator = Evaluator(model=model)

        test_loader = DataLoader(DummyDataset(num_samples=3), batch_size=1)
        metrics = evaluator.run(test_loader=test_loader, mean=[0.0], std=[1.0], mask=None)

        assert "psnr" in metrics
        assert "mae" in metrics

    def test_evaluator_run_with_structured_output(self):
        """评估阶段支持 ModelOutput"""
        model = DummyStructuredOutputModel()
        evaluator = Evaluator(model=model)

        test_loader = DataLoader(DummyDataset(num_samples=3), batch_size=1)
        metrics = evaluator.run(test_loader=test_loader, mean=[0.0], std=[1.0], mask=None)

        assert "psnr" in metrics
        assert "mae" in metrics

    def test_evaluator_run_inference_mode(self):
        """inference 模式只推理，不返回指标"""
        model = DummyModel()
        evaluator = Evaluator(model=model)

        test_loader = DataLoader(DummyInferenceDataset(num_samples=3), batch_size=1)
        metrics = evaluator.run(test_loader=test_loader, mean=[0.0], std=[1.0], mask=None)

        assert metrics == {}

    def test_evaluator_load_checkpoint(self, tmp_path):
        """测试检查点加载"""
        model = DummyModel()
        evaluator = Evaluator(model=model)

        # 创建并保存检查点
        ckpt_path = tmp_path / "test_checkpoint.pth"
        torch.save({"model": model.state_dict()}, ckpt_path)

        # 加载检查点
        evaluator.load_checkpoint(str(ckpt_path))

        assert evaluator.model.training is False

    def test_evaluator_load_checkpoint_weights_only(self, tmp_path):
        """测试仅权重格式的检查点加载"""
        model = DummyModel()
        evaluator = Evaluator(model=model)

        # 创建并保存仅权重的检查点
        ckpt_path = tmp_path / "test_weights.pth"
        torch.save(model.state_dict(), ckpt_path)

        # 加载检查点
        evaluator.load_checkpoint(str(ckpt_path))

        assert evaluator.model.training is False

    def test_evaluator_count_parameters(self):
        """测试参数计数"""
        model = DummyModel()
        evaluator = Evaluator(model=model)

        trainable = evaluator.count_parameters(trainable_only=True)
        total = evaluator.count_parameters(trainable_only=False)

        assert trainable > 0
        assert total >= trainable


class TestEvaluatorCallbacks:
    """Evaluator 回调测试"""

    def test_on_eval_begin(self):
        """测试 on_eval_begin 回调"""
        model = DummyModel()
        evaluator = Evaluator(model=model)

        # 调用回调
        evaluator.callbacks.on_eval_begin(evaluator)

        # 应该不报错
        assert True

    def test_on_eval_end(self):
        """测试 on_eval_end 回调"""
        model = DummyModel()
        evaluator = Evaluator(model=model)

        # 调用回调
        metrics = {"psnr": 30.0, "ssim": 0.9, "mae": 0.1, "max_mae": 0.5}
        evaluator.callbacks.on_eval_end(evaluator, metrics)

        # 应该不报错
        assert True


class TestMetricsCallback:
    """MetricsCallback 测试"""

    def test_metrics_callback_init(self):
        """测试初始化"""
        from src.callbacks import MetricsCallback

        callback = MetricsCallback(verbose=True)
        assert callback.verbose is True

    def test_metrics_callback_hooks(self):
        """测试回调钩子"""
        from src.callbacks import MetricsCallback

        callback = MetricsCallback(verbose=False)
        model = DummyModel()
        evaluator = Evaluator(model=model, callbacks=[callback])

        # 模拟评估过程
        callback.on_eval_begin(evaluator)
        callback.on_batch_end(
            evaluator, 1,
            sr=torch.randn(1, 1, 20, 20),
            hr=torch.randn(1, 1, 20, 20),
            filename="test.npy",
            metrics={"psnr": 30.0, "ssim": 0.9, "mae": 0.1}
        )
        callback.on_eval_end(evaluator, {"psnr": 30.0, "ssim": 0.9, "mae": 0.1, "max_mae": 0.5})

        assert callback.total_psnr == 30.0
        assert callback.num_batches == 1


class TestSaveResultsCallback:
    """SaveResultsCallback 测试"""

    def test_save_results_callback_init(self):
        """测试初始化"""
        from src.callbacks import SaveResultsCallback

        callback = SaveResultsCallback(save_dir="./test_results")
        assert callback.save_format == "npy"

    def test_save_results_callback_save(self, tmp_path):
        """测试保存结果"""
        from src.callbacks import SaveResultsCallback

        save_dir = tmp_path / "results"
        callback = SaveResultsCallback(save_dir=str(save_dir))

        model = DummyModel()
        evaluator = Evaluator(model=model, callbacks=[callback])

        # 模拟保存
        callback.on_eval_begin(evaluator)
        callback.on_batch_end(
            evaluator, 1,
            sr=torch.randn(1, 1, 20, 20),
            hr=torch.randn(1, 1, 20, 20),
            filename="test_sample.npy",
        )

        # 检查文件是否创建
        import os
        files = os.listdir(save_dir)
        assert len(files) == 1
        assert files[0] == "test_sample.npy"
