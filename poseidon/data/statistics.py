import numpy as np
import wandb
import xarray as xr

from pathlib import Path
from poseidon.utils import generate_paths
from typing import Optional, Sequence

# isort: split
from poseidon.config import POSEIDON_MASK


class PoseidonStatistics:
    r"""Compute the mean and standard deviation of a dataset along
    the temporal and horizontal spatial dimension using the
    one-pass algorithm."""

    def __init__(self):
        self.ds_sum = None
        self.ds_sum_squared = None
        self.total_count = None

    def update(self, dataset: xr.Dataset) -> None:
        r"""Update the current statistics with new data."""

        # Compute sums for the current dataset over the time dimension
        current_sum = dataset.sum(dim="time", skipna=True)
        current_sum_squared = (dataset**2).sum(dim="time", skipna=True)
        current_count = dataset.count(dim=["time", "latitude", "longitude"])

        # Initialize or update the statistics
        if self.ds_sum is None:
            self.ds_sum = current_sum
            self.ds_sum_squared = current_sum_squared
            self.total_count = current_count
        else:
            self.ds_sum += current_sum
            self.ds_sum_squared += current_sum_squared
            self.total_count += current_count

    def get_mean(self) -> xr.Dataset:
        r"""Compute the mean of the accumulated dataset."""
        return (self.ds_sum.sum(dim=["latitude", "longitude"], skipna=True) / self.total_count).fillna(0)

    def get_standard_deviation(self) -> xr.Dataset:
        r"""Compute the standard deviation of the accumulated dataset."""
        return np.sqrt(
            (
                self.ds_sum_squared.sum(dim=["latitude", "longitude"], skipna=True)
                / self.total_count
            )
            - (self.get_mean() ** 2)
        ).fillna(1)

def _preprocess_sample_for_statistics(
    dataset: xr.Dataset, mask: xr.Dataset, variables: Optional[Sequence[str]] = None
) -> xr.Dataset:
    r"""Filters an xarray dataset by retaining specific variables,
        removing unused variables, renaming dimensions, and setting
        appropriate attributes.

    Arguments:
        dataset: The input xarray dataset to be filtered.
        mask: Mask to apply to the dataset.
        variables: A list of variable names to retain.

    Returns:
        xr.Dataset: A new filtered dataset with renamed dimensions and updated attributes.
    """

    if variables is not None:
        dataset = dataset[variables]
    unused_vars = ["time_centered", "time_instant", "nav_lat", "nav_lon"]
    dataset = dataset.drop_vars(unused_vars)
    dataset = dataset.rename({
        "x": "longitude",
        "y": "latitude",
        "deptht": "level",
        "time_counter": "time",
    })

    # Broadcast the mask to the dataset time dimension
    mask = mask.expand_dims(dim="time", axis=0)
    mask = xr.concat([mask for _ in range(dataset.time.size)], dim="time")
    return dataset.where(mask["mask"] != 0)


def compute_statistics(
    output_path: Path,
    start_date: str,
    end_date: str,
    wandb_mode: str,
    variables: Optional[Sequence[str]] = None,
) -> None:
    r"""Computes mean and standard deviation for a dataset and saves the results to Zarr format.

    Args:
        output_path: Path where the output .zarr file will be saved.
        start_date: Start date for the data range in 'YYYY-MM' format.
        end_date: End date for the data range in 'YYYY-MM' format.
        wandb_mode: Mode for wandb (e.g., 'online', 'offline').
        variables: List of variable names to retain from the dataset.
    """

    wandb.init(project="Poseidon-Statistics", mode=wandb_mode)
    paths = generate_paths()
    stats_calculator = PoseidonStatistics()
    processing_data = False

    for date, path in paths.items():
        # Temporal Checking (1)
        if date == start_date:
            processing_data = True
        if not processing_data:
            continue

        # Load and preprocess dataset
        dataset = _preprocess_sample_for_statistics(
            dataset=xr.open_mfdataset(path, combine="by_coords", engine="netcdf4"),
            mask=xr.open_zarr(POSEIDON_MASK),
            variables=variables,
        )

        # Update statistics with current dataset
        stats_calculator.update(dataset)
        wandb.log({"Progress/Year": int(date[:4]), "Progress/Month": int(date[5:])})

        # Temporal Checking (2)
        if date == end_date:
            break

    # Compute and save the final statistics dataset
    mean = stats_calculator.get_mean().load()
    std = stats_calculator.get_standard_deviation().load()
    stats_ds = xr.concat(
        [mean, std],
        dim="statistic",
    )
    stats_ds = stats_ds.assign_coords(statistic=["mean", "std"])
    stats_ds.attrs.update({"Date (Start)": start_date, "Date (End)": end_date})
    stats_ds.to_zarr(output_path, mode="w")
    wandb.finish()