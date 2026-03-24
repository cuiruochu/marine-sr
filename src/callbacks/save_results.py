"""保存结果回调。"""

from pathlib import Path

import numpy as np

from src.core.callbacks import Callback


class SaveResultsCallback(Callback):
    def __init__(
        self,
        save_dir: str,
        save_format: str = "npy",
    ):
        self.save_dir = Path(save_dir)
        self.save_format = save_format

    def on_eval_begin(self, evaluator) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_batch_end(
        self,
        evaluator,
        batch_idx: int,
        logs: dict | None = None,
        sr=None,
        hr=None,
        filename: str | list[str] | tuple[str, ...] | None = None,
        mask=None,
        **kwargs,
    ) -> None:
        if sr is None or filename is None:
            return
        if sr.ndim != 4:
            raise ValueError("SaveResultsCallback 期望 sr shape 为 [B, C, H, W]")

        filenames = [filename] if isinstance(filename, str) else list(filename)
        if len(filenames) != sr.shape[0]:
            raise ValueError(f"filename 数量必须与 batch 大小一致，当前 filenames={len(filenames)}，batch={sr.shape[0]}")

        for sample_sr, sample_name in zip(sr, filenames, strict=False):
            sr_np = sample_sr.cpu().numpy()
            if mask is not None:
                mask_np = np.asarray(mask, dtype=bool)
                if sr_np.ndim == 2:
                    object_sr = sr_np.astype(object)
                    object_sr[~mask_np] = None
                else:
                    object_sr = sr_np.astype(object)
                    expanded_mask = np.broadcast_to(mask_np[None, ...], sr_np.shape)
                    object_sr[~expanded_mask] = None
                sr_np = object_sr
            save_path = self.save_dir / f"{Path(str(sample_name)).stem}.{self.save_format}"
            np.save(save_path, sr_np)
