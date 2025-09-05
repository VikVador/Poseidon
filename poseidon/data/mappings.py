r"""Tools for mapping data representations."""

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
from poseidon.data.const import DATASET_REGION, DATASET_VARIABLES


def from_tensor_to_indices(
    path: Path = PATH_DATA,
    region: Dict[str, slice] = DATASET_REGION,
    variables: Sequence[str] = DATASET_VARIABLES,
) -> Dict[str, Tuple[int, int]]:
    r"""Creates a mapping between variable and indices in a stacked tensor.

    Arguments:
        path: Path to the original dataset.
        region: Region on which the data is defined.
        variables: Variable present in the stacked tensor.
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
    path: Path = PATH_DATA,
    region: Dict[str, slice] = DATASET_REGION,
    variables: Sequence[str] = DATASET_VARIABLES,
) -> xr.Dataset:
    r"""Transform a trajectory into an Xarray dataset.

    Arguments:
        x: Trajectory (C, T, X, Y).
        path: Path to the original dataset.
        region: Region on which the data is defined.
        variables: Variable present in the stacked tensor.
    """
    assert 4 <= x.ndim < 6, "ERROR - Trajectory must have shape (C, T, X, Y)"
    while x.ndim < 5:
        x = x.unsqueeze(dim=0)

    # Extracting variables
    data_slices = {
        v: x[:, idx_start:idx_end]
        for v, (idx_start, idx_end) in from_tensor_to_indices(
            path=path,
            variables=variables,
            region=region,
        ).items()
    }

    # Creating dataset
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
