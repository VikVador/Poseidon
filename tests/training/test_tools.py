r"""Tests for the poseidon.training.tools module."""

import pytest
import torch

from torch import Tensor
from typing import Dict, Tuple

# isort: split
from poseidon.training.tools import (
    compute_blanket_indices,
    extract_blankets_in_trajectories,
)


@pytest.mark.parametrize(
    "trajectory_size, k, expected",
    [
        (
            5,
            1,
            {
                0: (0, 3),
                1: (0, 3),
                2: (1, 4),
                3: (2, 5),
                4: (2, 5),
            },
        ),
        (
            10,
            2,
            {
                0: (0, 5),
                1: (0, 5),
                2: (0, 5),
                3: (1, 6),
                4: (2, 7),
                5: (3, 8),
                6: (4, 9),
                7: (5, 10),
                8: (5, 10),
                9: (5, 10),
            },
        ),
    ],
)
def test_compute_blanket_indices(
    trajectory_size: int, k: int, expected: Dict[int, Tuple[int, int]]
):
    """Testing corectness of blankets position."""
    result = compute_blanket_indices(
        trajectory_size=trajectory_size,
        k=k,
    )
    assert (
        result == expected
    ), f"ERROR - Expected {expected}, but got {result} for trajectory_size={trajectory_size}, k={k}."


@pytest.mark.parametrize(
    "x, k, blankets_center_idx, expected_shape",
    [
        (torch.rand(2, 3, 10, 4, 4), 1, torch.tensor([2, 8]), (2, 3, 3, 4, 4)),
        (torch.rand(1, 2, 15, 5, 5), 2, torch.tensor([7]), (1, 2, 5, 5, 5)),
    ],
)
def test_extract_blankets_in_trajectories_shape(
    x: Tensor, k: int, blankets_center_idx: Tensor, expected_shape: Tuple[int, ...]
):
    """Testing that it returns tensor of correct shape."""
    result = extract_blankets_in_trajectories(x, k, blankets_center_idx)
    assert (
        result.shape == expected_shape
    ), f"ERROR - Expected shape {expected_shape}, but got {result.shape}"


@pytest.mark.parametrize(
    "k, T",
    [
        (1, 10),
        (2, 10),
        (3, 10),
        (1, 5),
    ],
)
def test_extract_blankets_in_trajectories_values(k, T):
    """Testing correctness of blanket extraction."""
    B, C, H, W = 3, 3, 4, 4
    x = torch.arange(B * C * T * H * W).reshape(B, C, T, H, W).float()
    for _ in range(25):
        blankets_center_idx = torch.randint(k, T - k, (B,))  # Ensure valid centers
        result = extract_blankets_in_trajectories(x, k, blankets_center_idx)
        for b in range(B):
            start, end = blankets_center_idx[b] - k, blankets_center_idx[b] + k + 1
            expected_blanket = x[b, :, start:end, :, :]
            assert torch.allclose(
                result[b], expected_blanket
            ), f"ERROR - Blanket extraction for batch {b} with center {blankets_center_idx[b].item()} and k={k}, T={T} is incorrect"
