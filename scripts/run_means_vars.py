r"""Script to compute monthly and seasonal mean/variance statistics."""

import argparse
import calendar
import pandas as pd
import torch
import xarray as xr

from dawgz import job, schedule
from pathlib import Path

# isort: split
from poseidon.config import PATH_DATA
from poseidon.data.const import (
    DATASET_DATES_TRAINING,
    DATASET_REGION,
    DATASET_VARIABLES,
)
from poseidon.data.datasets import PoseidonDataset

# fmt: off
MONTHLY_DATES = {
    calendar.month_name[m].lower(): [
        f"{m:02d}-{d:02d}"
        for d in range(1, calendar.monthrange(2000, m)[1] + 1)  # Use 2000 (leap year) for Feb
    ]
    for m in range(1, 13)
}

SEASONAL_DATES = {
    "winter": MONTHLY_DATES["december"]  + MONTHLY_DATES["january"] + MONTHLY_DATES["february"],
    "spring": MONTHLY_DATES["march"]     + MONTHLY_DATES["april"]   + MONTHLY_DATES["may"],
    "summer": MONTHLY_DATES["june"]      + MONTHLY_DATES["july"]    + MONTHLY_DATES["august"],
    "fall":   MONTHLY_DATES["september"] + MONTHLY_DATES["october"] + MONTHLY_DATES["november"],
}

def filter_dataset_by_mm_dd_list(dataset: xr.Dataset, mm_dd_list: list[str]) -> xr.Dataset:
    r"""Filter xarray dataset to dates matching MM-DD patterns."""

    # Convert MM-DD strings to (month, day) tuples for O(1) lookup
    target_dates = set()
    for mm_dd in mm_dd_list:
        month, day = map(int, mm_dd.split("-"))
        target_dates.add((month, day))

    # Create boolean mask using pandas timestamp accessor
    time_values = dataset.time.values
    mask = [(pd.Timestamp(t).month, pd.Timestamp(t).day) in target_dates for t in time_values]

    return dataset.isel(time=mask)


def compute_period_statistics(
    mm_dd_list: list[str],
    period_name: str,
    path_output: Path,
    subsampling: int = 1,
) -> None:
    r"""Compute mean and variance statistics for a specific period (month/season).

    Arguments:
        mm_dd_list: List of MM-DD date strings defining the period.
        period_name: Name for output files (e.g., "january", "winter").
        path_output: Base directory for output torch tensor files.
        subsampling: Subsampling factor (1 = all dates, 2 = every other date, etc.).
    """

    # Apply subsampling to date list
    mm_dd_list_subsampled = mm_dd_list[::subsampling]

    # Load full training dataset
    dataset = PoseidonDataset(
        path=PATH_DATA,
        date_start=DATASET_DATES_TRAINING[0],
        date_end=DATASET_DATES_TRAINING[1],
        trajectory_size=1,
        variables=DATASET_VARIABLES,
        region=DATASET_REGION,
    )

    # Access underlying xarray dataset
    xr_dataset = dataset.dataset

    # Filter by MM-DD dates
    filtered = filter_dataset_by_mm_dd_list(xr_dataset, mm_dd_list_subsampled)

    # Compute statistics (skipna=True is CRITICAL for handling land areas)
    mean_ds = filtered.mean(dim="time", skipna=True)
    var_ds  = filtered.var(dim="time", skipna=True)

    # Convert to torch tensors
    mean_tensor = torch.from_numpy(
        mean_ds.to_stacked_array(new_dim="z_total", sample_dims=("longitude", "latitude")).values
    ).float()

    var_tensor = torch.from_numpy(
        var_ds.to_stacked_array(new_dim="z_total", sample_dims=("longitude", "latitude")).values
    ).float()

    # Save to torch tensor files
    path_output_dir = Path(path_output)
    path_output_dir.mkdir(parents=True, exist_ok=True)
    mean_path = path_output_dir / f"mean_{period_name}_1998-2017.pt"
    var_path  = path_output_dir / f"var_{period_name}_1998-2017.pt"
    torch.save(mean_tensor, mean_path)
    torch.save(var_tensor, var_path)

    # Displaying information
    print(f"✓ {period_name}: {len(filtered.time)} samples (subsampling={subsampling})")
    print(f"  Mean: {mean_path}")
    print(f"  Var: {var_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute monthly and seasonal mean/variance statistics from training data.")

    parser.add_argument(
        "--path_output",
        "-o",
        type=str,
        required=True,
        help="Output directory for torch tensor files. Will create mean_* and var_* files.",
    )
    parser.add_argument(
        "--subsampling",
        "-s",
        type=int,
        default=4,
        help="Subsampling factor: 1=all dates, 2=every other date, etc. (default: 1).",
    )
    parser.add_argument(
        "--backend",
        "-b",
        type=str,
        default="slurm",
        choices=["slurm", "async"],
        help="Computation backend: 'slurm' for cluster, 'async' for local execution.",
    )

    args = parser.parse_args()

    # Create ordered lists for array job (12 months + 4 seasons = 16 jobs)
    all_periods, all_mm_dd_lists = [], []

    # Add monthly periods
    for month_name, mm_dd_list in MONTHLY_DATES.items():
        all_periods.append(month_name)
        all_mm_dd_lists.append(mm_dd_list)

    # Add seasonal periods
    for season_name, mm_dd_list in SEASONAL_DATES.items():
        all_periods.append(season_name)
        all_mm_dd_lists.append(mm_dd_list)

    SLURM_CONFIG = {
        "cpus":      1,
        "mem":       "16GB",
        "time":      "04:00:00",
        "account":   "bsmfc",
        "partition": "shared",
    }

    # Launching jobs with dawgz
    @job(array=len(all_periods), name="POSEIDON-MEANS-VARS", cpus=1, mem="16GB", time="04:00:00", account="bsmfc", partition="batch")
    def COMPUTE_STATS(i: int) -> None:
        compute_period_statistics(
            mm_dd_list  = all_mm_dd_lists[i],
            period_name = all_periods[i],
            path_output = args.path_output,
            subsampling = args.subsampling,
        )

    schedule(
        COMPUTE_STATS,
        name="POSEIDON-MEANS-VARS",
        export="ALL",
        backend=args.backend,
    )
