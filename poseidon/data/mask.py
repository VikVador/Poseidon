r"""Tools for handling masks."""

import torch
import xarray as xr

from pathlib import Path
from torch import Tensor
from typing import (
    Dict,
    Sequence,
)

# isort: split
from poseidon.config import PATH_DATA
from poseidon.data.const import DATASET_REGION, DATASET_VARIABLES


def generate_trajectory_mask(
    trajectory_size: int,
    path: Path = PATH_DATA,
    region: Dict[str, slice] = DATASET_REGION,
    variables: Sequence[str] = DATASET_VARIABLES,
) -> Tensor:
    r"""Creates a boolean trajectory mask.

    Notes

    Arguments:
        trajectory_size: Trajectory dimension (T).
        path: Path to the original dataset.
        region: Region on which the data is defined.
        variables: Variable present in the stacked tensor.

    Returns:
        mask: (1, C, T, X, Y).
    """

    mask = xr.open_zarr(path)[variables].isel(time=0).isel(**region)
    mask = mask.to_stacked_array(new_dim="z_total", sample_dims=("longitude", "latitude")).transpose("z_total", ...)
    mask = torch.as_tensor(mask.load().data.copy())
    mask = ~torch.isnan(mask) * 1.0
    return mask.unsqueeze(1).repeat(1, trajectory_size, 1, 1).unsqueeze(0)
