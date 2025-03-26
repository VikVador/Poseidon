r"""A collection of tools designed for training module."""

import torch

from einops import rearrange
from torch import Tensor
from typing import Dict, Tuple


def compute_blanket_indices(trajectory_size: int, k: int) -> Dict[int, Tuple[int, int]]:
    r"""Determines the position of each blanket in a trajectory.

    Arguments:
        trajectory_size: Number of time steps in trajectory
        k: Number of neighbors on each side of the blanket.

    Returns:
        Dictionary with blankets center as keys and start/end positions as values.
    """
    return {
        i: (0, 2 * k + 1)
        if i <= k
        else (trajectory_size - 2 * k - 1, trajectory_size)
        if i >= trajectory_size - k
        else (i - k, i + k + 1)
        for i in range(trajectory_size)
    }


def extract_blankets_in_trajectories(x: Tensor, k: int, blankets_center_idx: Tensor) -> Tensor:
    r"""Extracts blankets from tensor given indices of their centers.

    Arguments:
        x: Input tensor (B, C, T, X, Y).
        k: Number of neighbors on each side of the blanket.
        blankets_center_idx: Position of blankets center (B).

    Returns:
        Blankets from input tensor (B, C, K, X, Y)
    """
    _, _, T, _, _ = x.shape

    blankets_indices = compute_blanket_indices(
        trajectory_size=T,
        k=k,
    )

    x_blankets = torch.stack(
        [
            x[b, :, blankets_indices[c.item()][0] : blankets_indices[c.item()][1], :, :]
            for b, c in enumerate(blankets_center_idx)
        ],
        dim=0,
    )

    return x_blankets


def extract_random_blankets(x: Tensor, k: int, flatten: bool = True) -> Tensor:
    r"""Extracts random blankets from tensor and flattens it.

    Arguments:
        x: Input tensor (B, C, T, X, Y).
        k: Number of neighbors on each side of the blanket.
        flatten: Whether to flatten the output tensor or not.

    Returns:
        Blankets from input tensor.
    """

    B, _, T, _, _ = x.shape

    assert (
        T > 2 * k + 1
    ), f"ERROR - Trajectory size must be greater than blanket size ({T} > {2 * k + 1})"

    x = extract_blankets_in_trajectories(
        x=x,
        k=k,
        blankets_center_idx=torch.randint(0, T, (B,)),
    )

    x = rearrange(x, "B ... -> B (...)") if flatten else x

    return x
