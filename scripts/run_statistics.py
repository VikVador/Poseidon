r"""Script to compute statistics over a dataset."""

import argparse

from dawgz import Job, schedule

# isort: split
from poseidon.config import POSEIDON_STAT
from poseidon.data.statistics import compute_statistics
from poseidon.data.const import (
    DATASET_TRAINING_DATE_END,
    DATASET_TRAINING_DATE_START,
    DATASET_VARIABLES,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-compute statistics over a Black Sea dataset.")

    parser.add_argument(
        "--output_path",
        "-o",
        type=str,
        default=POSEIDON_STAT,
        help="Output .zarr file path.",
    )
    parser.add_argument(
        "--date_start",
        "-ds",
        type=str,
        default=DATASET_TRAINING_DATE_START,
        help="Start date (YYYY-MM).",
    )
    parser.add_argument(
        "--date_end",
        "-de",
        type=str,
        default=DATASET_TRAINING_DATE_END,
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
        help="Computation backend.",
    )

    args = parser.parse_args()

    def dawgz_statistics():
        compute_statistics(
            output_path=args.output_path,
            start_date=args.date_start,
            end_date=args.date_end,
            wandb_mode="online" if args.use_wandb else "disabled",
            variables=args.variables,
        )

    schedule(
        Job(
            dawgz_statistics,
            cpus=4,
            mem="128GB",
            name="POSEIDON-STATS",
            time="00-00:10:00",
            account="bsmfc",
            partition="batch",
        ),
        name="POSEIDON-STATS",
        export="ALL",
        backend=args.backend,
    )
