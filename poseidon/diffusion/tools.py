r"""Diffusion - Helper tools to preprocess data for the diffusion pipeline."""

import torch

from typing import Tuple


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
        x: Input tensor (B, T, C, H, W).
        blanket_idx: Tuple containing the starting and ending indices of the blankets.

    Returns:
        blankets: Blanket tensor of shape (B, 2k+1, C, H, W).
    """
    idx_start, idx_end = blanket_idx
    blankets = [x[i, start:end, :, :, :] for i, (start, end) in enumerate(zip(idx_start, idx_end))]
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
