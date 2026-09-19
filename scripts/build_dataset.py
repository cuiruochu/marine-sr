from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.data_pipeline.pipeline import build_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an SR dataset from ERA5 NetCDF files. "
            "The maximum selected scale determines the shared HR crop."
        )
    )
    parser.add_argument(
        "--oper-nc",
        type=Path,
        required=True,
        help="NetCDF file containing the u10 and v10 variables.",
    )
    parser.add_argument(
        "--wave-nc",
        type=Path,
        required=True,
        help="NetCDF file containing the mwd, mwp, and swh variables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output dataset directory, for example: dataset-2022/test.",
    )
    parser.add_argument(
        "--scales",
        type=int,
        nargs="+",
        required=True,
        help="One or more upscaling factors, for example: 2 4.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    args = parse_args()
    build_dataset(
        oper_nc=args.oper_nc,
        wave_nc=args.wave_nc,
        output_dir=args.output_dir,
        scales=args.scales,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
