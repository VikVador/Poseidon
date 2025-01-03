r"""Tests for the poseidon.diffusion.loss module."""

import pytest
import random
import torch

# isort: split
from poseidon.diffusion.loss import PoseidonLoss

# Generating random dimensions for testing
INPUT_B, INPUT_D = (random.randint(2, 16) for _ in range(2))


@pytest.fixture
def fake_input():
    """Fixture to provide a noisy inputs tensor for testing."""
    x_t = torch.randn(INPUT_B, INPUT_D)
    x_t_denoised = torch.randn(INPUT_B, INPUT_D)
    return x_t, x_t_denoised


@pytest.fixture
def fake_noise():
    """Fixture to provide a noise level tensor for testing."""
    sigma = torch.randn(INPUT_B, 1)
    sigma.requires_grad = True
    sigma = sigma.to(dtype=torch.float32)
    return sigma


def test_poseidon_loss_shapes(fake_input, fake_noise):
    """Testing PoseidonLoss with valid input shapes."""
    (x_t, x_t_denoised), sigma_t = fake_input, fake_noise

    loss = PoseidonLoss(
        x_t,
        x_t_denoised,
        sigma_t,
    )

    assert loss.ndim == 0, f"ERROR - Expected scalar loss, got {loss.ndim}-D tensor."
    assert loss.dtype == torch.float32, f"ERROR - Expected dtype torch.float32, got {loss.dtype}."
