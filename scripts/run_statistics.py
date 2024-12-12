r"""Script to perform statistical analysis of a dataset."""

import argparse

from dawgz import job, schedule
from functools import partial

# isort: split
from poseidon.data.const import (
    DATASET_DATES_TRAINING,
    DATASET_VARIABLES,
    DATASET_VARIABLES_CLIPPING,
)
from poseidon.data.statistics import compute_statistics
from poseidon.data.tools import assert_date_format

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute statistics of a dataset.")

    parser.add_argument(
        "--path_output",
        "-o",
        type=str,
        required=True,
        help="Output .zarr file path.",
    )
    parser.add_argument(
        "--date_start",
        "-ds",
        type=str,
        default=DATASET_DATES_TRAINING[0],
        help="Start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--date_end",
        "-de",
        type=str,
        default=DATASET_DATES_TRAINING[1],
        help="End date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--variables",
        "-v",
        type=str,
        nargs="+",
        default=DATASET_VARIABLES,
        help="List of variables for which compute statistics.",
    )
    parser.add_argument(
        "--use_wandb",
        "-w",
        action="store_true",
        help="Use Weights & Biases for logging advancement.",
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
    assert_date_format(args.date_start)
    assert_date_format(args.date_end)

    dawgz_statistics = partial(
        compute_statistics,
        args.path_output,
        args.date_start,
        args.date_end,
        "online" if args.use_wandb else "disabled",
        args.variables,
        DATASET_VARIABLES_CLIPPING,
    )

    schedule(
        job(
            dawgz_statistics,
            cpus=8,
            mem="128GB",
            name="POSEIDON-STATS",
            time="01-00:00:00",
            account="bsmfc",
            partition="shared",
        ),
        name="POSEIDON-STATS",
        export="ALL",
        backend=args.backend,
    )
