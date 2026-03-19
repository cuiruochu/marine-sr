"""随机数状态工具。"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """统一设置 Python、NumPy、PyTorch 随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    """捕获当前进程的随机数状态。"""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": _serialize_numpy_state(np.random.get_state()),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    """恢复随机数状态。"""
    if not state:
        return

    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(_deserialize_numpy_state(state["numpy"]))
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _serialize_numpy_state(state: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "bit_generator": state[0],
        "state": state[1].tolist(),
        "pos": int(state[2]),
        "has_gauss": int(state[3]),
        "cached_gaussian": float(state[4]),
    }


def _deserialize_numpy_state(state: dict[str, Any]) -> tuple[Any, ...]:
    return (
        state["bit_generator"],
        np.array(state["state"], dtype=np.uint32),
        int(state["pos"]),
        int(state["has_gauss"]),
        float(state["cached_gaussian"]),
    )
