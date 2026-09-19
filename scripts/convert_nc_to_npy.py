from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

DATASET_DIR = Path(r"C:\projects\paper\dataset-2022")
OUTPUT_DIR = DATASET_DIR / "test"

NC_TO_VARIABLES = {
    DATASET_DIR / "data_stream-wave_stepType-instant.nc": ("mwd", "mwp", "swh"),
}


def main() -> None:
    with xr.open_dataset(DATASET_DIR / "data_stream-oper_stepType-instant.nc") as dataset:
        output_dir = OUTPUT_DIR / "wind-raw"
        output_dir.mkdir(parents=True, exist_ok=True)

        values = np.hypot(
            np.asarray(dataset["u10"].values, dtype=np.float32),
            np.asarray(dataset["v10"].values, dtype=np.float32),
        )
        values = np.nan_to_num(values, nan=0.0)
        times = dataset["valid_time"].dt.strftime("%Y-%m-%dT%H").values

        for index, (time_label, frame) in enumerate(zip(times, values, strict=True)):
            output_path = output_dir / f"{time_label}.npy"
            np.save(output_path, frame.astype(np.float32))
            print(f"[{index + 1}/{len(times)}] {output_path}")

    for nc_path, variable_names in NC_TO_VARIABLES.items():
        with xr.open_dataset(nc_path) as dataset:
            for variable_name in variable_names:
                output_dir = OUTPUT_DIR / f"{variable_name}-raw"
                output_dir.mkdir(parents=True, exist_ok=True)

                values = np.asarray(dataset[variable_name].values, dtype=np.float32)
                values = np.nan_to_num(values, nan=0.0)
                times = dataset["valid_time"].dt.strftime("%Y-%m-%dT%H").values

                for index, (time_label, frame) in enumerate(zip(times, values, strict=True)):
                    output_path = output_dir / f"{time_label}.npy"
                    np.save(output_path, frame.astype(np.float32))
                    print(f"[{index + 1}/{len(times)}] {output_path}")


if __name__ == "__main__":
    main()
