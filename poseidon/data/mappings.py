r"""A collection of tools handling mapping between data representation."""

import xarray as xr

from pathlib import Path
from torch import Tensor
from typing import (
    Dict,
    Sequence,
    Tuple,
)

# isort: split
from poseidon.config import PATH_DATA


def from_tensor_to_indices(
    variables: Sequence[str],
    region: Dict[str, slice],
    path: Path = PATH_DATA,
) -> Dict[str, Tuple[int, int]]:
    r"""Determine variables position in a stacked tensor.

    Arguments:
        variables: Variable present in the stacked tensor.
        region: Region used to extract the data from original dataset.
        path: Path to the original dataset.

    Returns:
        Mapping dictionary [variable, (pos_start, pos_end)]
    """

    dataset = xr.open_zarr(path)[variables]
    idx_start, mapping = 0, {}
    if isinstance(region["level"], list):
        total_levels = len(region["level"])
    else:
        total_levels = region["level"].stop - region["level"].start

    for v in dataset:
        idx_end = idx_start + (total_levels if "level" in dataset[v].dims else 1)
        mapping[v] = (idx_start, idx_end)
        idx_start = idx_end

    return mapping


def from_tensor_to_xarray(
    x: Tensor,
    variables: Sequence[str],
    region: Dict[str, slice],
    path: Path = PATH_DATA,
) -> xr.Dataset:
    r"""Transform a (batch of) stacked tensor into an :class:`Xarray dataset`.

    Arguments:
        x: Input tensor (C, T, X, Y).
        variables: Variable present in the stacked tensor.
        region: Region used to extract the data from original dataset.
        path: Path to the original dataset.
    """
    assert 4 <= x.ndim < 6, "ERROR - Input tensor must have shape (C, T, X, Y)"
    while x.ndim < 5:
        x = x.unsqueeze(dim=0)

    # Extracting data associated to each variable
    data_slices = {
        v: x[:, idx_start:idx_end]
        for v, (idx_start, idx_end) in from_tensor_to_indices(
            path=path,
            variables=variables,
            region=region,
        ).items()
    }

    # Creating Xarray dataset
    data_arrays = []
    for v, data in data_slices.items():
        data_array = xr.DataArray(
            data=data,
            dims=("batch", "level", "trajectory", "latitude", "longitude"),
            name=v,
        )

        if data_array.shape[1] == 1:
            data_array = data_array.squeeze(dim="level")
        data_arrays.append(data_array)

    return xr.merge(data_arrays)
