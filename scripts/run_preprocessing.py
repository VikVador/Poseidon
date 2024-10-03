r"""Scripts - Preprocess the data (standardization, fixing nans, ...)."""

import argparse

from dawgz import Job, schedule

# isort: split
from poseidon.config import (
    POSEIDON_DATA,
    POSEIDON_STAT,
)
from poseidon.data.const import (
    DATASET_TESTING_DATE_END,
    DATASET_TRAINING_DATE_START,
    DATASET_VARIABLES,
    DATASET_VARIABLES_CLIPPING,
    DATASET_VARIABLES_SURFACE,
)
from poseidon.data.preprocessing import compute_preprocessed_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess a Black Sea dataset.")

    parser.add_argument(
        "--output_path",
        "-o",
        type=str,
        default=POSEIDON_DATA,
        help="Output .zarr file path.",
    )
    parser.add_argument(
        "--statistics_path",
        "-s",
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
        default=DATASET_TESTING_DATE_END,
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

    def dawgz_preprocessing_data():
        compute_preprocessed_dataset(
            output_path=args.output_path,
            statistics_path=args.statistics_path,
            start_date=args.date_start,
            end_date=args.date_end,
            wandb_mode="online" if args.use_wandb else "disabled",
            variables=args.variables,
            variables_clipping=DATASET_VARIABLES_CLIPPING,
            variables_surface=DATASET_VARIABLES_SURFACE,
        )

    schedule(
        Job(
            dawgz_preprocessing_data,
            cpus=16,
            mem="256GB",
            name="POSEIDON-PREPROCESSING",
            time="05-00:00:00",
            account="bsmfc",
            partition="shared",
        ),
        name="POSEIDON-PREPROCESSING",
        export="ALL",
        backend=args.backend,
    )
