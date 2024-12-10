r"""Tests for the poseidon.network.normalization module."""

import pytest
import random
import torch

# isort: split
from poseidon.network.normalization import LayerNorm


@pytest.fixture
def sample_data():
    """Generate a random tensor with random dimensions for testing."""
    B, X, Y, SCALING = (random.randint(50, 100) for _ in range(4))
    x = torch.randn(B, X, Y) * SCALING
    return x


@pytest.mark.parametrize("dim", [0, 1, (0, 1)])
@pytest.mark.parametrize("eps", [1e-7, 1e-5, 1e-3])
def test_layer_norm_basic_functionality(sample_data, dim, eps):
    """Testing basic functionality of LayerNorm: mean should be near 0 and variance near 1."""
    # Create LayerNorm instance
    layer_norm = LayerNorm(dim=dim, eps=eps)

    # Apply normalization
    x = sample_data
    y = layer_norm(x)
    mean, variance = (
        y.mean(dim=dim if isinstance(dim, tuple) else (dim,)),
        y.var(dim=dim if isinstance(dim, tuple) else (dim,), unbiased=False),
    )

    # Assertions
    assert y.shape == x.shape

    assert torch.allclose(
        mean, torch.zeros_like(mean), atol=1e-3
    ), f"ERROR - Mean mismatch: {mean}"

    assert torch.allclose(
        variance, torch.ones_like(variance), atol=1e-1
    ), f"ERROR - Variance mismatch: {variance}"


@pytest.mark.parametrize("dim", [0, 1])
def test_layer_norm_eps_behavior(sample_data, dim):
    """Testing numerical stability with different epsilon values."""
    x = sample_data * 1e-8
    small_eps = LayerNorm(dim=dim, eps=1e-10)(x)
    large_eps = LayerNorm(dim=dim, eps=1e-3)(x)

    assert torch.all(
        small_eps.abs() >= large_eps.abs()
    ), "ERROR - Numerical instability observed with small epsilon value"


def test_layer_norm_edge_cases():
    """Testing edge cases for LayerNorm."""

    # Test with a tensor with a large number of dimensions
    B, X, Y, Z, W, T, SCALING = (random.randint(1, 10) for _ in range(7))
    x = torch.randn(B, X, Y, Z, W, T) * SCALING
    layer_norm = LayerNorm(dim=(1, 2))
    y = layer_norm(x)
    assert (
        y.shape == x.shape
    ), f"ERROR - Failed with large number of dimensions: expected {x.shape}, got {y.shape}"

    # Test with a large tensor
    B, X, Y, SCALING = (random.randint(250, 750) for _ in range(4))
    x = torch.randn(B, X, Y) * SCALING
    layer_norm = LayerNorm(dim=1)
    y = layer_norm(x)
    assert y.shape == x.shape, "ERROR - Failed on large tensor"
