r"""Mappings between data representations."""

import xarray as xr

from pathlib import Path
from torch import Tensor
from typing import Dict, Sequence, Tuple

# isort: split
from poseidon.config import PATH_DATA
from poseidon.data.const import DATASET_REGION, DATASET_VARIABLES


def from_tensor_to_indices(
    path: Path = PATH_DATA,
    variables: Sequence[str] = DATASET_VARIABLES,
    region: Dict[str, slice] = DATASET_REGION,
) -> Dict[str, Tuple[int, int]]:
    r"""Determine variables position in a stacked tensor.

    Arguments:
        path: Path to the original .zarr dataset containing the variables.
        variables: Variable present in the stacked torch tensor.
        region: Region of interest to extract from the dataset.

    Returns:
        Mapping dictionary [variable, (pos_start, pos_end)]
    """

    dataset = xr.open_zarr(path)[variables]

    # Creation of the mapping
    idx_start, mapping, total_levels = 0, {}, region["level"].stop
    for v in dataset:
        idx_end = idx_start + (total_levels if "level" in dataset[v].dims else 1)
        mapping[v] = (idx_start, idx_end)
        idx_start = idx_end

    return mapping


def from_tensor_to_xarray(
    x: Tensor,
    path: Path = PATH_DATA,
    variables: Sequence[str] = DATASET_VARIABLES,
    region: Dict[str, slice] = DATASET_REGION,
) -> xr.Dataset:
    r"""Transform a stacked tensor to an xarray dataset.

    Arguments:
        x: Input tensor (*, Z, T, X, Y).
        path: Path to the original .zarr dataset containing the variables.
        variables: Variable present in the stacked torch tensor.
        region: Region of interest to extract from the dataset.
    """
    assert x.ndim >= 4, "ERROR - Input tensor must have shape (*, Z, T, X, Y)"
    while x.ndim < 5:
        x = x.unsqueeze(dim=0)

    data_slices = {
        v: x[:, idx_start:idx_end]
        for v, (idx_start, idx_end) in from_tensor_to_indices(
            path=path,
            variables=variables,
            region=region,
        ).items()
    }

    # Creating xarray dataset
    data_arrays = []
    for v, data in data_slices.items():
        data_array = xr.DataArray(
            data=data,
            dims=("batch", "level", "trajectory", "latitude", "longitude"),
            name=v,
        )

        if data_array.shape[1] == 1:  # Surface variables
            data_array = data_array.squeeze(dim="level")
        data_arrays.append(data_array)

    return xr.merge(data_arrays)
