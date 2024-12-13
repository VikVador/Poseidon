r"""A collection of tools designed for diffusion module."""

import torch
import xarray as xr

from einops import rearrange
from pathlib import Path
from torch import Tensor
from typing import Dict

# isort: split
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
