r"""Data - Tools to perfom specific preprocessing steps over a dataset."""

import wandb
import xarray as xr

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

# isort: split
from poseidon.config import PATH_MASK
from poseidon.utils import generate_paths


def cliping(
    dataset: xr.Dataset, variables_clipping: Optional[Dict[str, Tuple[int, int]]] = None
) -> xr.Dataset:
    r"""Helper function to clip values in a dataset.

    Arguments:
        dataset: Dataset whose variables must be clipped.
        variables_clipping: Dictionary of variables and their respective min and max values.
    """
    if variables_clipping is None:
        return dataset
    for variable, (min_value, max_value) in variables_clipping.items():
        dataset[variable] = dataset[variable].clip(min=min_value, max=max_value)
    return dataset


def preprocess_sample(
    dataset: xr.Dataset,
    mask: xr.Dataset,
    variables: Optional[Sequence[str]] = None,
    variables_clipping: Optional[Dict[str, Tuple[int, int]]] = None,
) -> xr.Dataset:
    r"""Filters an xarray dataset by retaining specific variables,
        removing unused variables, renaming dimensions, and setting
        appropriate attributes.

    Arguments:
        dataset: The input xarray dataset to be filtered.
        mask: Mask to apply to the dataset.
        variables: A list of variable names to retain.
        variables_clipping: Dictionary of variables and their respective min and max values.
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
    dataset = cliping(dataset=dataset, variables_clipping=variables_clipping)

    # Broadcast the mask to the dataset time dimension
    mask = mask.expand_dims(dim="time", axis=0)
    mask = xr.concat([mask for _ in range(dataset.time.size)], dim="time")
    return dataset.where(mask["mask"] != 0)


def compute_preprocessed_dataset(
    output_path: Path,
    statistics_path: Path,
    start_date: str,
    end_date: str,
    wandb_mode: str,
    variables: Optional[Sequence[str]] = None,
    variables_clipping: Optional[Dict[str, Tuple[int, int]]] = None,
    variables_surface: Optional[Sequence[str]] = None,
) -> None:
    r"""Preprocess raw data using feature extraction, clipping, standardization, ...

    Arguments:
        output_path: Path where the output .zarr file will be saved.
        statistics_path: Path to the statistics file.
        start_date: Start date for the data range in 'YYYY-MM' format.
        end_date: End date for the data range in 'YYYY-MM' format.
        wandb_mode: Mode for wandb (e.g., 'online', 'offline').
        variables: List of variable names to retain from the dataset.
        variables_clipping: Dictionary of variables and their respective min and max values.
        variables_surface: List of variables to retain only the surface level.
    """

    # Initialization
    wandb.init(project="Poseidon-Preprocessing", mode=wandb_mode)
    mask = xr.open_zarr(PATH_MASK)
    paths = generate_paths()
    stat = xr.open_zarr(statistics_path)
    mean = stat.sel(statistic="mean").load()
    std = stat.sel(statistic="std").load()
    preprocessing_state = False

    for date, path in paths.items():
        # Xarray is badly designed for appending data (bug if nothing initialy exists)
        xarray_mode = "a"

        # -- Temporal Check (1) --
        if date == start_date:
            preprocessing_state = True
            xarray_mode = "w"
        if not preprocessing_state:
            continue

        # -- Preprocessing --
        dataset = xr.open_mfdataset(path)
        dataset = preprocess_sample(dataset, mask, variables, variables_clipping)
        dataset = (dataset - mean) / std
        for var in variables_surface:
            dataset[var] = dataset[var].isel(level=0)

        # Fixing chuncks for better performance
        dataset = dataset.chunk({"time": 1})
        dataset.to_zarr(
            output_path, mode=xarray_mode, append_dim="time"
        ) if xarray_mode == "a" else dataset.to_zarr(output_path, mode=xarray_mode)
        dataset.close()
        wandb.log({"Progress/Year": int(date[:4]), "Progress/Month": int(date[5:])})

        # -- Temporal Check (2) --
        if date == end_date:
            break

    wandb.finish()
