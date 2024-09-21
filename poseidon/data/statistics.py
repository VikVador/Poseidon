import numpy as np
import wandb
import xarray as xr

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

# isort: split
from poseidon.config import POSEIDON_MASK
from poseidon.data.preprocessing import preprocess_sample
from poseidon.utils import generate_paths


class PoseidonStatistics:
    r"""Compute the mean and standard deviation of a dataset along
    the temporal and horizontal spatial dimension using the
    one-pass algorithm."""

    def __init__(self):
        self.mu = None
        self.mu_squared = None
        self.total_count = None
        self.eps = 1e-32

    def update(self, dataset: xr.Dataset) -> None:
        r"""Update the current statistics with new data."""

        # Computing current batch statistics
        dimensions = ["time", "latitude", "longitude"]
        batch_mu = dataset.mean(dim=dimensions, skipna=True)
        batch_mu_squared = (dataset**2).mean(dim=dimensions, skipna=True)
        batch_count = dataset.count(dim=dimensions)

        # Initialization
        if self.mu is None:
            self.mu = batch_mu
            self.mu_squared = batch_mu_squared
            self.total_count = batch_count

        # Update
        else:
            const_1 = self.total_count / (self.total_count + batch_count)
            const_2 = batch_count / (self.total_count + batch_count)
            self.mu = const_1 * self.mu + const_2 * batch_mu
            self.mu_squared = const_1 * self.mu_squared + const_2 * batch_mu_squared
            self.total_count += batch_count

        # Loading into memory
        self.mu.load()
        self.mu_squared.load()
        self.total_count.load()

    def get_mean(self) -> xr.Dataset:
        r"""Compute the mean of the accumulated dataset."""
        return self.mu.fillna(0)

    def get_standard_deviation(self) -> xr.Dataset:
        r"""Compute the standard deviation of the accumulated dataset."""
        return np.sqrt(self.mu_squared - self.mu**2).fillna(1).clip(min=self.eps)


def compute_statistics(
    output_path: Path,
    start_date: str,
    end_date: str,
    wandb_mode: str,
    variables: Optional[Sequence[str]] = None,
    variables_clipping: Optional[Dict[str, Tuple[int, int]]] = None,
) -> None:
    r"""Computes mean and standard deviation for a dataset and saves the results to Zarr format.

    Args:
        output_path: Path where the output .zarr file will be saved.
        start_date: Start date for the data range in 'YYYY-MM' format.
        end_date: End date for the data range in 'YYYY-MM' format.
        wandb_mode: Mode for wandb (e.g., 'online', 'offline').
        variables: List of variable names to retain from the dataset.
        variables_clipping: Dictionary of variables and their respective min and max values.

    """

    wandb.init(project="Poseidon-Statistics", mode=wandb_mode)
    paths = generate_paths()
    stats_calculator = PoseidonStatistics()
    processing_data = False

    for date, path in paths.items():
        # -- Temporal Check (1) --
        if date == start_date:
            processing_data = True
        if not processing_data:
            continue

        # Load and preprocess dataset
        dataset = preprocess_sample(
            dataset=xr.open_mfdataset(path, combine="by_coords", engine="netcdf4"),
            mask=xr.open_zarr(POSEIDON_MASK),
            variables=variables,
            variables_clipping=variables_clipping,
        )

        # Update statistics with current dataset
        stats_calculator.update(dataset)
        wandb.log({"Progress/Year": int(date[:4]), "Progress/Month": int(date[5:])})
        dataset.close()

        # -- Temporal Check (2) --
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
