from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .nc_reader import OPER_VARIABLES, WAVE_VARIABLES, read_time_labels, read_variable_frames, validate_nc_variables
from .transforms import bicubic_downsample, combine_wind_speed, encode_mwd, trim_to_scale, wave_first_mask

LOGGER = logging.getLogger(__name__)


def build_dataset(
    *,
    oper_nc: Path,
    wave_nc: Path,
    output_dir: Path,
    scales: Iterable[int],
) -> None:
    selected_scales = sorted(set(scales))
    if not selected_scales:
        raise ValueError("At least one scale is required")
    if any(scale <= 1 for scale in selected_scales):
        raise ValueError("All scales must be greater than 1")

    validate_nc_variables(oper_nc, OPER_VARIABLES)
    validate_nc_variables(wave_nc, WAVE_VARIABLES)

    oper_times = read_time_labels(oper_nc)
    wave_times = read_time_labels(wave_nc)
    if len(oper_times) != len(wave_times):
        raise ValueError(
            f"Oper and wave NC files have different sample counts: "
            f"{len(oper_times)} vs {len(wave_times)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    max_scale = max(selected_scales)

    LOGGER.info("Processing wind from %s", oper_nc)
    u10 = read_variable_frames(oper_nc, "u10")
    v10 = read_variable_frames(oper_nc, "v10")
    wind = combine_wind_speed(u10, v10)
    del u10, v10
    wind = wind[:, None, :, :]
    save_variable_frames(
        name="wind",
        frames=trim_to_scale(wind, max_scale),
        time_labels=oper_times,
        output_dir=output_dir,
        scales=selected_scales,
    )
    del wind

    for variable_name in WAVE_VARIABLES:
        LOGGER.info("Processing %s from %s", variable_name, wave_nc)
        frames = read_variable_frames(wave_nc, variable_name)
        masks = wave_first_mask(frames)
        if variable_name == "mwd":
            frames = np.stack([encode_mwd(frame) for frame in frames], axis=0)
        else:
            frames = np.nan_to_num(frames, nan=0.0).astype(np.float32, copy=False)
            frames = frames[:, None, :, :]

        save_variable_frames(
            name=variable_name,
            frames=trim_to_scale(frames, max_scale),
            time_labels=wave_times,
            output_dir=output_dir,
            scales=selected_scales,
            masks=trim_to_scale(masks, max_scale),
        )
        del frames, masks


def save_variable_frames(
    *,
    name: str,
    frames: np.ndarray,
    time_labels: list[str],
    output_dir: Path,
    scales: list[int],
    masks: np.ndarray | None = None,
) -> None:
    if len(frames) != len(time_labels):
        raise ValueError(f"{name}: frame count and time count differ: {len(frames)} vs {len(time_labels)}")
    if masks is not None and masks.ndim != 2:
        raise ValueError(f"{name}: mask must have shape (H, W), got {tuple(masks.shape)}")

    for scale in scales:
        variable_dir = output_dir / name / f"x{scale}"
        hr_dir = variable_dir / "hr"
        lr_dir = variable_dir / "lr"
        hr_dir.mkdir(parents=True, exist_ok=True)
        lr_dir.mkdir(parents=True, exist_ok=True)

        if masks is not None:
            np.save(variable_dir / "mask.npy", masks)

        for index, time_label in enumerate(time_labels):
            hr = frames[index]
            lr = bicubic_downsample(hr, scale)
            output_name = f"{time_label}.npy"
            np.save(hr_dir / output_name, hr.astype(np.float32, copy=False))
            np.save(lr_dir / output_name, lr.astype(np.float32, copy=False))

        LOGGER.info(
            "Saved %s x%s: %d samples, HR=%s, LR=%s",
            name,
            scale,
            len(time_labels),
            tuple(hr.shape),
            tuple(lr.shape),
        )
