r"""Tests for the poseidon.diffusion.schedulers module."""

import pytest
import random
import torch

# isort: split
from poseidon.diffusion.schedulers import PoseidonNoiseScheduler

# Generating random parameters for testing
SIGMA_MIN, SIGMA_MAX, SPREAD = (random.uniform(1, 49), random.uniform(50, 100), 2)


@pytest.mark.parametrize("scheduler", [PoseidonNoiseScheduler()])
def test_forward_output_dtype(scheduler):
    """Testing output tensor dtype consistency."""
    t = torch.rand((10, 1))
    sigma_t = scheduler(t)
    assert sigma_t.dtype == torch.float32, "ERROR - Expected torch.float32 sigma_t dtype"


@pytest.mark.parametrize("batch_size", [1, 10, 100])
@pytest.mark.parametrize("scheduler", [PoseidonNoiseScheduler()])
def test_forward_variable_sizes(scheduler, batch_size):
    """Testing output shape for different batch sizes."""
    t = torch.rand((batch_size, 1))
    sigma_t = scheduler(t)
    assert sigma_t.shape == (
        batch_size,
        1,
    ), f"ERROR - Shape mismatch: expected ({batch_size},1), got {sigma_t.shape}"


@pytest.mark.parametrize("scheduler", [PoseidonNoiseScheduler()])
def test_edge_cases(scheduler):
    """Testing boundary conditions (t=0 and t=1)."""
    t_edges = torch.tensor([0.0, 1.0])
    sigma_t = scheduler(t_edges)

    assert not torch.isnan(sigma_t).any(), "ERROR - NaN values detected in edge cases"
    assert (sigma_t[0] < sigma_t[1]).all(), "ERROR - Noise schedule should be increasing"
