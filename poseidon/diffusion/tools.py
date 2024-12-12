r"""Diffusion backbone helping conditionning data."""

import torch
import xarray as xr

from einops import rearrange
from pathlib import Path
from torch import Tensor
from typing import Dict, Tuple

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


# ------ TO BE REWORKED ------


def time_tokenizer(data: torch.Tensor) -> torch.Tensor:
    r"""Tokenizes a tensor containing temporal information.

    Arguments:
        data: Time tensor of shape (batch_size, 3), i.e. month, day, and hour.
    """
    months = data[:, 0] - 1
    days = data[:, 1] - 1
    hour_mapping = {6: 0, 12: 1, 18: 2, 24: 3}
    hours = torch.tensor([hour_mapping[int(hour)] for hour in data[:, 2]], dtype=torch.long)
    return torch.stack([months, days, hours], dim=1)


def extract_blankets_in_trajectories(
    x: torch.Tensor, blanket_idx: Tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    r"""Extract various blankets from a given batch of trajectories.

    Arguments:
        x: Input tensor (B, C, T, H, W).
        blanket_idx: Tuple containing the starting and ending indices of the blankets.

    Returns:
        blankets: Blanket tensor of shape (B, C, 2k+1, H, W).
    """
    idx_start, idx_end = blanket_idx
    blankets = [x[i, :, start:end] for i, (start, end) in enumerate(zip(idx_start, idx_end))]
    return torch.stack(blankets, dim=0)


def compute_blanket_indices(
    indices: torch.Tensor, k: int, trajectory_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Given a set of random indexes, determines the blanket position around it.

    Arguments:
        indices: Random indices for the blanket's center (B).
        k: Number of neighbors to consider on each side to define the "blanket".
        trajectory_size: Total length of the trajectory.

    Returns:
        idx_start: Starting indices for the blankets (B).
        idx_end: Ending indices for the blankets (B).
        idx_state: Index of the center state in the blanket (B).
    """
    idx_start = torch.clip(indices - k, min=0)
    idx_end = torch.clip(indices + k + 1, max=trajectory_size)
    pad_start = torch.clip(k - indices, min=0)
    pad_end = torch.clip(indices + k + 1 - trajectory_size, min=0)
    idx_start -= pad_end
    idx_end += pad_start
    idx_state = indices - idx_start
    return idx_start, idx_end, idx_state
