r"""Tests for the poseidon.diffusion.noise module."""

import pytest
import random
import torch

# isort: split
from poseidon.diffusion.noise import (
    PoseidonNormalLogNoiseSchedule,
    PoseidonUniformLogNoiseSchedule,
)

# Generating random noise values
MU, SIGMA, SIGMA_MIN, SIGMA_MAX = (
    torch.randn(1).item(),
    torch.randn(1).item(),
    random.uniform(1, 49),
    random.uniform(50, 100),
)


def test_initialization():
    """Testing the initialization of the noise schedules."""

    scheduler_normal, scheduler_uniform = (
        PoseidonNormalLogNoiseSchedule(
            mu=MU,
            sigma=SIGMA,
        ),
        PoseidonUniformLogNoiseSchedule(
            sigma_min=SIGMA_MIN,
            sigma_max=SIGMA_MAX,
        ),
    )

    assert torch.allclose(
        scheduler_normal.mu, torch.tensor(MU)
    ), f"ERROR (NormalLog) - Mean value is incorrect: {scheduler_normal.mu.item()}"

    assert torch.allclose(
        scheduler_normal.sigma, torch.tensor(SIGMA)
    ), f"ERROR (NormalLog) - Standard deviation value is incorrect: {scheduler_normal.sigma.item()}"

    assert torch.allclose(
        torch.exp(scheduler_uniform.log_sigma_min), torch.tensor(SIGMA_MIN)
    ), f"ERROR - (UniformLog) Sigma min is incorrect: {scheduler_uniform.log_sigma_min.item()}"

    assert torch.allclose(
        torch.exp(scheduler_uniform.log_sigma_max), torch.tensor(SIGMA_MAX)
    ), f"ERROR - (UniformLog) Sigma max is incorrect: {scheduler_uniform.log_sigma_max.item()}"


@pytest.mark.parametrize(
    "scheduler", [PoseidonNormalLogNoiseSchedule(), PoseidonUniformLogNoiseSchedule()]
)
def test_forward_output_dtype(scheduler):
    """Testing that the forward method returns the correct dtype."""
    output = scheduler(10)
    assert (
        output.dtype == torch.float32
    ), f"ERROR - Expected dtype torch.float32, got {output.dtype}."


@pytest.mark.parametrize("batch_size", [1, 10, 100])
@pytest.mark.parametrize(
    "scheduler", [PoseidonNormalLogNoiseSchedule(), PoseidonUniformLogNoiseSchedule()]
)
def test_forward_variable_sizes(scheduler, batch_size):
    """Testing forward method with varying batch sizes."""
    output = scheduler(batch_size)
    assert output.shape == (
        batch_size,
        1,
    ), f"ERROR - Expected shape {(batch_size, 1)} for size {batch_size}, got {output.shape}."
