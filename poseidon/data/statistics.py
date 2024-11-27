r"""Tools to compute statistics of a dataset."""

import numpy as np
import wandb
import xarray as xr

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

# isort: split
from poseidon.config import PATH_MASK
from poseidon.data.preprocessing import dataset_preprocessing
from poseidon.data.tools import generate_paths


class PoseidonStatistics:
    r"""Compute the mean and standard deviation of a dataset along
    the temporal and horizontal spatial dimension using the
    one-pass algorithm."""

    def __init__(self):
        self.mu = None
        self.mu_squared = None
        self.total_count = None
        self.eps = 1e-8

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
    path_output: Path,
    date_start: str,
    date_end: str,
    wandb_mode: str,
    variables: Optional[Sequence[str]] = None,
    clipping: Optional[Dict[str, Tuple[int, int]]] = None,
) -> None:
    r"""Computes statistics of a Black Sea dataset and saves them.

    Statistics:
        The mean and standard deviation are computed for each variable. Additionally,
        statistics are computed independently for each level, reflecting the fact that
        dynamics at different levels may differ (e.g., surface vs. higher atmospheric levels).

    Arguments:
        path_output: Path where the output .zarr file will be saved.
        date_start: Start date for the data range in 'YYYY-MM' format.
        date_end: End date for the data range in 'YYYY-MM' format.
        wandb_mode: Wether to use Weights & Biases for logging or not.
        variables: Variable for which statistics are computed.
        clipping: Physical limits for each variable, i.e. domain of definition.
    """
    wandb.init(project="Poseidon-Statistics", mode=wandb_mode)

    # Initialization
    paths = generate_paths()
    mask = xr.open_zarr(PATH_MASK).load()
    stats_calculator = PoseidonStatistics()

    for date, path in paths.items():
        # Temporal Filtering
        if date < date_start or date_end < date:
            continue

        # Do not load until variables are selected
        dataset = dataset_preprocessing(
            dataset=xr.open_mfdataset(path, combine="by_coords", engine="netcdf4"),
            mask=mask,
            variables=variables,
            clipping=clipping,
        )

        stats_calculator.update(dataset)
        dataset.close()

        wandb.log({"Progress/Year": int(date[:4]), "Progress/Month": int(date[5:])})

    # Load and save the final statistics dataset
    dataset_statistics = xr.concat(
        [
            stats_calculator.get_mean().load(),
            stats_calculator.get_standard_deviation().load(),
        ],
        dim="statistic",
    )
    dataset_statistics = dataset_statistics.assign_coords(statistic=["mean", "std"])
    dataset_statistics.attrs.update({"Date (Start)": date_start, "Date (End)": date_end})
    dataset_statistics.to_zarr(path_output, mode="w")

    wandb.finish()
