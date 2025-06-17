r"""A collection of tools designed for diagnostics module."""

import torch
import xarray as xr

from datetime import datetime, timedelta
from typing import Dict, Sequence, Tuple

# isort: split
from poseidon.config import PATH_DATA, PATH_NOWCASTS, PATH_STAT


def previous_day(date_str: str) -> str:
    """Returns the previous day as a string in 'YYYY-MM-DD' format.

    Note:
        If the input date is March 1st of a leap year, it returns February 28th.

    Arguments:
        date_str: Date string in 'YYYY-MM-DD' format.

    Returns:
        Previous day's date in 'YYYY-MM-DD' format.
    """
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    if date_obj.month == 3 and date_obj.day == 1:
        year = date_obj.year
        is_leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        if is_leap:
            return f"{year}-02-28"
    return (date_obj - timedelta(days=1)).strftime("%Y-%m-%d")


def _load_and_process_zarr(
    path: str,
    variables: Sequence[str],
    region: Dict[str, Tuple[int, int]],
) -> torch.Tensor:
    """Loads and processes a Zarr dataset, returning a stacked torch.Tensor.

    Arguments:
        path: Path to the Zarr dataset.
        variables: List of variable names to extract.
        region: Dictionary specifying region slicing.

    Returns:
        Stacked tensor of selected variables and region.
    """
    ds = xr.open_zarr(path).sel(statistic="mean").isel(**region)[variables].load()
    if "ssh" in variables:
        ds = ds.assign(ssh=ds["ssh"].isel(level=0))
    return torch.as_tensor(
        ds.to_stacked_array(new_dim="z_total", sample_dims=("longitude", "latitude"))
        .transpose("z_total", ...)
        .data.copy()
    )


def get_nowcasts(
    date: str,
    variables: Sequence[str],
    region: Dict[str, Tuple[int, int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generates baseline nowcasts of the Black Sea.

    Arguments:
        date: Date for which to generate the nowcast in 'YYYY-MM-DD' format.
        variables: List of variable names to extract.
        region: Dictionary specifying region slicing.

    Returns:
        Nowcast tensors:
            - Daily climatology
            - Yearly climatology
            - Persistent nowcast (last available observation)
    """
    print(PATH_NOWCASTS)
    daily_path = f"{PATH_NOWCASTS}/climatology_daily/{date[5:10]}_local_mean_std.zarr"
    ds_daily = _load_and_process_zarr(daily_path, variables, region)
    ds_yearly = _load_and_process_zarr(
        f"{PATH_NOWCASTS}/climatology_yearly.zarr", variables, region
    )

    ds = xr.open_zarr(PATH_DATA)
    ds_stats = xr.open_zarr(PATH_STAT)
    ds_persistent = ds * ds_stats.sel(statistic="std") + ds_stats.sel(statistic="mean")
    ds_persistent = ds_persistent.sel(time=date).isel(time=0, **region)[variables].load()
    if "ssh" in variables:
        ds_persistent = ds_persistent.assign(ssh=ds_persistent["ssh"].isel(level=0))
    ds_persistent_tensor = torch.as_tensor(
        ds_persistent.to_stacked_array(new_dim="z_total", sample_dims=("longitude", "latitude"))
        .transpose("z_total", ...)
        .data.copy()
    )

    return ds_daily, ds_yearly, ds_persistent_tensor
