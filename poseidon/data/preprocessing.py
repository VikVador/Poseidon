r"""Tools to perform dataset preprocessing."""

import wandb
import xarray as xr

from pathlib import Path
from typing import (
    Dict,
    Optional,
    Sequence,
    Tuple,
)

# isort: split
from poseidon.config import PATH_MASK
from poseidon.data.tools import generate_paths


def dataset_clipping(
    dataset: xr.Dataset,
    clipping: Optional[Dict[str, Tuple[int, int]]] = None,
) -> xr.Dataset:
    r"""Bounds variables to their physical limits.

    Information
        Numerical errors can lead to values outside the physical limits of the variables.

    Arguments:
        dataset: Dataset whose variables must be physically bounded.
        clipping: Dictionary where keys are variable names, and values are tuples defining
                  the minimum and maximum bounds for each variable. If both are `None`,
                  no clipping is applied for that variable.
    """

    if clipping:
        # Loop through subset of variables and their physical limits
        for variable, (min_value, max_value) in clipping.items():
            # Unbounded domain is not handled by clip (issue)
            if min_value is None and max_value is None:
                continue

            # Clip the variable to its physical limits
            dataset[variable] = dataset[variable].clip(
                min=min_value,
                max=max_value,
            )

    return dataset


def dataset_preprocessing(
    dataset: xr.Dataset,
    mask: Optional[xr.Dataset] = None,
    variables: Optional[Sequence[str]] = None,
    clipping: Optional[Dict[str, Tuple[int, int]]] = None,
) -> xr.Dataset:
    r"""Preprocess a raw Black Sea xarray dataset of any dimension.

    Preprocessing:
        1. Drop unused variables.
        2. Rename coordinates.
        3. Physical bounds clipping.
        4. Masking land values.

    Arguments:
        dataset: Unprocessed input xarray dataset.
        mask: Mask to apply to the dataset (with same dimensions).
        variables: List of variable names to retain from the original dataset.
        clipping: Dictionary where keys are variable names, and values are tuples defining
                  the minimum and maximum bounds for each variable. If both are `None`,
                  no clipping is applied for that variable.
    """

    if variables is not None:
        dataset = dataset[variables]

    dataset = dataset.drop_vars([
        "time_centered",
        "time_instant",
        "nav_lat",
        "nav_lon",
    ])

    dataset = dataset.rename({
        "x": "longitude",
        "y": "latitude",
        "deptht": "level",
        "time_counter": "time",
    })

    dataset = dataset_clipping(
        dataset=dataset,
        clipping=clipping,
    )

    if mask:
        # Broadcast the mask to the dataset time dimension
        mask = mask.expand_dims(dim="time", axis=0)
        mask = xr.concat([mask for _ in range(dataset.time.size)], dim="time")
        dataset = dataset.where(mask["mask"] != 0)

    return dataset


def compute_preprocessing(
    path_output: Path,
    path_statistics: Path,
    date_start: str,
    date_end: str,
    wandb_mode: str,
    variables: Optional[Sequence[str]] = None,
    clipping: Optional[Dict[str, Tuple[int, int]]] = None,
    variables_surface: Optional[Sequence[str]] = None,
) -> None:
    r"""Launch a preprocessing pipeline for a Black Sea dataset.

    Arguments:
        path_output: Path where the output .zarr file will be saved.
        path_statistics: Path to the (pre-computed) statistics file.
        date_start: Start date for the data range in 'YYYY-MM' format.
        date_end: End date for the data range in 'YYYY-MM' format.
        wandb_mode: Wether to use Weights & Biases for logging or not.
        variables: Variable for which are retained in the final dataset.
        variables_surface: Variables only defined at the surface.
        clipping: Physical limits for each variable, i.e. domain of definition.
    """
    wandb.init(project="Poseidon-Preprocessing", mode=wandb_mode)

    paths = generate_paths()
    mask = xr.open_zarr(PATH_MASK)
    stat = xr.open_zarr(path_statistics)
    mean, std = (
        stat.sel(statistic="mean").load(),
        stat.sel(statistic="std").load(),
    )

    for date, path in paths.items():
        # Temporal Filtering
        if date < date_start or date_end < date:
            continue

        dataset = xr.open_mfdataset(path)
        dataset = dataset_preprocessing(dataset, mask, variables, clipping)
        dataset = (dataset - mean) / std

        if variables_surface:
            for var in variables_surface:
                if var in dataset:
                    dataset[var] = dataset[var].isel(level=0)

        # Chunk dataset for performance
        dataset = dataset.chunk({"time": 1})

        # Determine the mode for writing to Zarr (solve Xarray bug)
        xarray_mode = "w" if date == date_start else "a"

        # Write dataset to output .zarr file
        dataset.to_zarr(
            path_output, mode=xarray_mode, append_dim="time"
        ) if xarray_mode == "a" else dataset.to_zarr(path_output, mode=xarray_mode)
        dataset.close()

        wandb.log({"Progress/Year": int(date[:4]), "Progress/Month": int(date[5:])})

    wandb.finish()
