r"""Tests for the poseidon.diffusion.noise module."""

import pytest
import random
import torch

# isort: split
from poseidon.diffusion.noise import PoseidonNoiseSchedule


def test_default_initialization():
    """Testing the default initialization of the noise schedule."""
    scheduler = PoseidonNoiseSchedule()
    assert torch.allclose(
        scheduler.mu, torch.tensor(-1.2)
    ), f"ERROR - Default mu value is incorrect: {scheduler.mu.item()}"
    assert torch.allclose(
        scheduler.sigma, torch.tensor(1.2)
    ), f"ERROR - Default sigma value is incorrect: {scheduler.sigma.item()}"


def test_random_initialization():
    """Testing random initialization of mu and sigma."""
    mu, sigma = random.uniform(-2.0, 2.0), random.uniform(0.1, 2.0)
    scheduler = PoseidonNoiseSchedule(mu=mu, sigma=sigma)
    assert torch.allclose(
        scheduler.mu, torch.tensor(mu)
    ), f"ERROR - Custom mu value is incorrect: {scheduler.mu.item()}"
    assert torch.allclose(
        scheduler.sigma, torch.tensor(sigma)
    ), f"ERROR - Custom sigma value is incorrect: {scheduler.sigma.item()}"


def test_forward_output_dtype():
    """Testing that the forward method returns the correct dtype."""
    scheduler = PoseidonNoiseSchedule()
    output = scheduler(10)
    assert (
        output.dtype == torch.float32
    ), f"ERROR - Expected dtype torch.float32, got {output.dtype}."


@pytest.mark.parametrize("batch_size", [1, 10, 100])
def test_forward_variable_sizes(batch_size):
    """Testing forward method with varying input sizes."""
    scheduler = PoseidonNoiseSchedule()
    output = scheduler(batch_size)
    assert output.shape == (
        batch_size,
        1,
    ), f"ERROR - Expected shape {(batch_size, 1)} for size {batch_size}, got {output.shape}."
