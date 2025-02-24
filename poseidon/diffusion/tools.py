r"""A collection of tools designed for diffusion module."""

import torch
import xarray as xr

from einops import rearrange
from pathlib import Path
from torch import Tensor
from typing import Dict, Sequence, Tuple

# isort: split
from poseidon.config import PATH_MASKV
from poseidon.network.encoding import SineEncoding


def generate_encoded_mesh(
    path: Path,
    features: int,
    region: Dict,
) -> Tensor:
    """Generates a sin/cos encoded mesh of a Black Sea region.

    Arguments:
        path: Path to the Black Sea mesh.
        features: Even number of sin/cos embeddings (F).
        region: Region of interest to extract from the dataset.

    Returns:
        Tensor: Encoded mesh (X Y (Mesh Levels F)).
    """

    mesh_data = xr.open_zarr(path).isel(**region).load()

    # Stack mesh variables into a single tensor
    mesh = torch.stack(
        [torch.from_numpy(mesh_data[v].values) for v in mesh_data.variables],
        dim=0,
    )
    mesh = rearrange(
        SineEncoding(features).forward(mesh),
        "... X Y F -> X Y (F ...)",
    )

    return mesh.to(dtype=torch.float32)


def get_mask_variables(
    variables: Sequence[str], region: Dict[str, Tuple[int, int]], blanket_size: int
) -> torch.Tensor:
    r"""Creates a mask for selected physical variables as a single tensor.

    Arguments:
        variables: Variable names to retain from the preprocessed dataset.
        region: Region of interest to extract from the dataset.
        blanket_size: Total number of elements in a blanket.

    Returns:
        Tensor of shape (1, z_total, blanket_size, latitude, longitude).
    """
    # Load and preprocess the mask dataset
    mask = xr.open_zarr(PATH_MASKV)[variables].isel(**region)
    mask = mask.to_stacked_array(
        new_dim="z_total", sample_dims=("longitude", "latitude")
    ).transpose("z_total", ...)

    # Convert to PyTorch tensor and apply transformations
    mask = torch.as_tensor(mask.load().data.copy())
    return mask.unsqueeze(1).repeat(1, blanket_size, 1, 1).unsqueeze(0)
