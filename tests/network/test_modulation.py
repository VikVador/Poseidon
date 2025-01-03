r"""Tests for the poseidon.network.modulation module."""

import pytest
import random
import torch

# isort: split
from poseidon.network.modulation import Modulator

# Generating random dimensions for testing
BATCH, CHANNELS, MOD_FEATURES = (random.randint(1, 32) for _ in range(3))


@pytest.fixture
def modulator():
    """Initialize a Modulator instance."""
    return Modulator(channels=CHANNELS, mod_features=MOD_FEATURES, spatial=2)


@pytest.fixture
def fake_modulating_vector():
    """Fixture to provide a modulating tensor for testing."""
    return torch.randn(BATCH, MOD_FEATURES)


def test_modulator_output_shape(modulator, fake_modulating_vector):
    """Testing modulating vectors shape."""
    output = modulator(fake_modulating_vector)
    batch_size = fake_modulating_vector.size(0)
    channels = modulator.ada_zero[-2].out_features // 3
    spatial_dims = (1,) * modulator.ada_zero[-1].pattern.count("1")
    expected_shape = (3, batch_size, channels, *spatial_dims)
    assert (
        output.shape == expected_shape
    ), f"ERROR - Output shape {output.shape} does not match expected shape {expected_shape}."


def test_modulator_weight_initialization(modulator):
    """Testing weights range of last linear layer."""
    layer = modulator.ada_zero[-2]
    avg_weight = torch.mean(torch.abs(layer.weight))
    tolerance = 5e-3
    assert (
        avg_weight < 1e-2 + tolerance
    ), "ERROR - Weight initialization does not match the expected scaling (~1e-2)."
    assert torch.all(
        layer.weight.abs() < 1e-2 + tolerance
    ), "ERROR - Outlier(s) in weight initialization."


def test_modulator_forward_consistency(modulator, fake_modulating_vector):
    """Testing the forward pass consistency."""
    output1 = modulator(fake_modulating_vector)
    output2 = modulator(fake_modulating_vector)
    assert output1.shape == output2.shape, "ERROR - Inconsistent output shapes."
    assert torch.allclose(
        output1, output2
    ), "ERROR - Forward pass is not consistent for the same input."


def test_modulator_differentiability(modulator, fake_modulating_vector):
    """Testing if the Modulator is differentiable."""
    output = modulator(fake_modulating_vector)
    loss = output.sum()
    loss.backward()
    for name, param in modulator.named_parameters():
        assert param.grad is not None, f"ERROR - Gradient not computed for {name} : {param}"


def test_modulator_invalid_input_shape(modulator):
    """Testing if the Modulator raises an error for invalid input shapes."""
    invalid_input = torch.randn(BATCH, MOD_FEATURES * 2)
    with pytest.raises(RuntimeError, match="shapes cannot be multiplied"):
        modulator(invalid_input)
