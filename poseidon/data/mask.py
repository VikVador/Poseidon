r"""A collection of tools designed to handle masks."""

import torch
import xarray as xr

from pathlib import Path
from torch import Tensor
from typing import (
    Dict,
    Sequence,
    Tuple,
)

# isort: split
from poseidon.config import PATH_MASKV


def generate_trajectory_mask(
    variables: Sequence[str],
    region: Dict[str, Tuple[int, int]],
    trajectory_size: int,
    path: Path = PATH_MASKV,
) -> Tensor:
    r"""Creates a boolean mask whose dimensions match preprocessed trajectory sample.

    Information:
        From dataloader, trajectory samples have shape (B, C, T, X, Y) where
        along the C dimension, variables are stacked. This function creates a
        mask tensor of shape (1, C, T, X, Y) where C dimension contains the
        corresponding variables.

    Arguments:
        variables: Variable names to retain from the dataset.
        region: Region of interest to extract from the dataset.
        trajectory_size: Total number of masks needed to cover the entire trajectory (T).
        path: Path to custom mask dataset (each physical variable and its corresponding mask).

    Returns:
        Bool mask tensor (1, C, T, X, Y).
    """
    mask = xr.open_zarr(path)[variables].isel(**region)
    mask = mask.to_stacked_array(
        new_dim="z_total", sample_dims=("longitude", "latitude")
    ).transpose("z_total", ...)
    mask = torch.as_tensor(mask.load().data.copy())
    return mask.unsqueeze(1).repeat(1, trajectory_size, 1, 1).unsqueeze(0)
