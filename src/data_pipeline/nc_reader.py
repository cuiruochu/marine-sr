from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

OPER_VARIABLES = ("u10", "v10")
WAVE_VARIABLES = ("mwd", "mwp", "swh")


def read_variable_frames(nc_path: Path, variable_name: str) -> np.ndarray:
    with xr.open_dataset(nc_path) as dataset:
        values = np.asarray(dataset[variable_name].values, dtype=np.float32)

    if values.ndim != 3:
        raise ValueError(
            f"{variable_name} in {nc_path} must have shape "
            f"(valid_time, latitude, longitude), got {tuple(values.shape)}"
        )
    return values


def read_time_labels(nc_path: Path) -> list[str]:
    with xr.open_dataset(nc_path) as dataset:
        labels = dataset["valid_time"].dt.strftime("%Y-%m-%dT%H").values.tolist()

    if not labels:
        raise ValueError(f"No valid_time values in {nc_path}")
    if len(labels) != len(set(labels)):
        raise ValueError(f"Duplicate valid_time values in {nc_path}")
    return labels


def validate_nc_variables(nc_path: Path, variable_names: tuple[str, ...]) -> None:
    if not nc_path.exists():
        raise FileNotFoundError(f"NC file does not exist: {nc_path}")

    with xr.open_dataset(nc_path) as dataset:
        missing = [name for name in variable_names if name not in dataset]

    if missing:
        raise ValueError(f"{nc_path} missing required variables: {', '.join(missing)}")

