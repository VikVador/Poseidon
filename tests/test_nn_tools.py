r"""Tests for the poseidon.network.tools module."""

import pytest
import random
import torch

# isort: split
from poseidon.network.tools import (
    reshape,
    unshape,
)


@pytest.fixture
def sample_data():
    """Fake sample tensor and modulation vector with random dimensions."""
    B, C, T, H, W, F = (random.randint(1, 10) for _ in range(6))
    x = torch.randn(B, C, T, H, W)
    mod = torch.randn(B, F)
    shape = (B, C, T, H, W)
    return x, mod, shape


def test_reshape_space(sample_data):
    """Testing reshaping with spatial hiding."""
    x, mod, shape = sample_data
    B, C, T, H, W = shape
    reshaped_x, reshaped_mod, original_shape = reshape(
        hide="space",
        x=x,
        mod=mod,
    )

    assert reshaped_x.shape == (B * H * W, C, T)
    assert reshaped_mod.shape == (B * H * W, mod.shape[1])
    assert original_shape == (B, C, T, H, W)


def test_reshape_time(sample_data):
    """Testing reshaping with temporal hiding."""
    x, mod, shape = sample_data
    B, C, T, H, W = shape
    reshaped_x, reshaped_mod, original_shape = reshape(
        hide="time",
        x=x,
        mod=mod,
    )

    assert reshaped_x.shape == (B * T, C, H, W)
    assert reshaped_mod.shape == (B * T, mod.shape[1])
    assert original_shape == (B, C, T, H, W)


def test_reshape_without_mod(sample_data):
    """Testing reshaping without modulation vector."""
    x, _, shape = sample_data
    B, C, T, H, W = shape
    reshaped_x, reshaped_mod, original_shape = reshape(
        hide="space",
        x=x,
    )

    assert reshaped_x.shape == (B * H * W, C, T)
    assert reshaped_mod is None
    assert original_shape == (B, C, T, H, W)


def test_unshape_space(sample_data):
    """Testing unshaping from spatial hiding."""
    x, _, shape = sample_data
    reshaped_x, _, _ = reshape("space", x)
    unshaped_x = unshape("space", reshaped_x, shape)

    assert unshaped_x.shape == shape
    assert torch.allclose(x, unshaped_x)


def test_unshape_time(sample_data):
    """Testing unshaping from temporal hiding."""
    x, _, shape = sample_data
    reshaped_x, _, _ = reshape("time", x)
    unshaped_x = unshape("time", reshaped_x, shape)

    assert unshaped_x.shape == shape
    assert torch.allclose(x, unshaped_x)


def test_invalid_reshape_mode(sample_data):
    """Testing reshaping with an invalid mode."""
    x, _, _ = sample_data
    with pytest.raises(ValueError):
        reshape("invalid", x)


def test_invalid_unshape_mode(sample_data):
    """Testing unshaping with an invalid mode."""
    x, _, shape = sample_data
    with pytest.raises(ValueError):
        unshape("invalid", x, shape)
