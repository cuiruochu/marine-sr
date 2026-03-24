"""
随机状态管理
"""

import random
from typing import Any, Dict

import numpy as np
import torch


def seed_everything(seed: int):
    """设置所有随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> Dict[str, Any]:
    """捕获当前随机状态。"""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _coerce_torch_rng_state(state: Any) -> torch.Tensor:
    """将 RNG state 规范为 CPU 上的 torch.uint8 Tensor。"""
    if isinstance(state, torch.Tensor):
        return state.detach().to(device="cpu", dtype=torch.uint8).flatten()
    if isinstance(state, np.ndarray):
        return torch.as_tensor(state, dtype=torch.uint8, device="cpu").flatten()
    return torch.tensor(list(state), dtype=torch.uint8, device="cpu")


def restore_rng_state(state: Dict[str, Any]):
    """恢复随机状态。"""
    if state is None:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_coerce_torch_rng_state(state["torch"]))
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([_coerce_torch_rng_state(item) for item in state["cuda"]])
