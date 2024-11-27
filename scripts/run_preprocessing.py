r"""Script to launch a data preprocessing pipeline."""

import argparse

from dawgz import job, schedule
from functools import partial

# isort: split
from poseidon.config import (
    PATH_STAT,
)
from poseidon.data.const import (
    DATASET_DATES_TEST,
    DATASET_DATES_TRAINING,
    DATASET_VARIABLES,
    DATASET_VARIABLES_CLIPPING,
    DATASET_VARIABLES_SURFACE,
)
from poseidon.data.preprocessing import compute_preprocessing

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess a dataset.")

    parser.add_argument(
        "--path_output",
        "-o",
        type=str,
        required=True,
        help="Output .zarr file path.",
    )
    parser.add_argument(
        "--path_statistics",
        "-s",
        type=str,
        default=PATH_STAT,
        help="Pre-computed statistics .zarr file path.",
    )
    parser.add_argument(
        "--date_start",
        "-ds",
        type=str,
        default=DATASET_DATES_TRAINING[0],
        help="Start date (YYYY-MM).",
    )
    parser.add_argument(
        "--date_end",
        "-de",
        type=str,
        default=DATASET_DATES_TEST[1],
        help="End date (YYYY-MM).",
    )
    parser.add_argument(
        "--variables",
        "-v",
        type=str,
        nargs="+",
        default=DATASET_VARIABLES,
        help="Variables to keep in dataset.",
    )
    parser.add_argument(
        "--use_wandb",
        "-w",
        action="store_true",
        help="Use Weights & Biases for logging advancement of the computation.",
    )
    parser.add_argument(
        "--backend",
        "-b",
        type=str,
        default="slurm",
        choices=["slurm", "async"],
        help="Computation backend, 'slurm' for cluster-based scheduling and 'async' for local execution.",
    )

    args = parser.parse_args()

    dawgz_preprocessing = partial(
        compute_preprocessing,
        args.path_output,
        args.path_statistics,
        args.date_start,
        args.date_end,
        "online" if args.use_wandb else "disabled",
        args.variables,
        DATASET_VARIABLES_CLIPPING,
        DATASET_VARIABLES_SURFACE,
    )

    schedule(
        job(
            dawgz_preprocessing,
            cpus=8,
            mem="128GB",
            name="POSEIDON-PREPROCESSING",
            time="02-00:00:00",
            account="bsmfc",
            partition="shared",
        ),
        name="POSEIDON-PREPROCESSING",
        export="ALL",
        backend=args.backend,
    )
