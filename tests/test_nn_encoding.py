r"""Tests for the poseidon.network.encoding module."""

import random
import pytest
import torch

# isort: split
from poseidon.network.encoding import SineEncoding

# Generating random dimensions for testing
BATCH, INPUT_DIM, FEATURES = (random.randint(1, 32) for _ in range(3))
FEATURES *= 2  # SineEncoding requires an even number of features
OMEGA = 1e3


@pytest.fixture
def sine_encoding():
    """Initialize a SineEncoding instance."""
    return SineEncoding(features=FEATURES, omega=OMEGA)


@pytest.fixture
def fake_input_vector():
    """Fixture to provide a random input tensor for testing."""
    t = torch.randn(BATCH, INPUT_DIM)
    t.requires_grad = True
    return t


def test_sine_encoding_output_shape(sine_encoding, fake_input_vector):
    """Testing the output shape of SineEncoding."""
    output = sine_encoding(fake_input_vector)
    expected_shape = (*fake_input_vector.shape, FEATURES)
    assert (
        output.shape == expected_shape
    ), f"ERROR - Output shape {output.shape} does not match expected shape {expected_shape}."


def test_sine_encoding_value_range(sine_encoding, fake_input_vector):
    """Testing the value range of SineEncoding output."""
    output = sine_encoding(fake_input_vector)
    assert torch.all(
        (output >= -1) & (output <= 1)
    ), "ERROR - SineEncoding output contains values outside the range [-1, 1]."


def test_sine_encoding_forward_consistency(sine_encoding, fake_input_vector):
    """Testing the forward pass consistency of SineEncoding."""
    output1 = sine_encoding(fake_input_vector)
    output2 = sine_encoding(fake_input_vector)
    assert output1.shape == output2.shape, "ERROR - Inconsistent output shapes."
    assert torch.allclose(
        output1, output2
    ), "ERROR - Forward pass is not consistent for the same input."


def test_sine_encoding_differentiability(sine_encoding, fake_input_vector):
    """Testing if SineEncoding is differentiable."""
    output = sine_encoding(fake_input_vector)
    loss = output.sum()
    loss.backward()
    for name, param in sine_encoding.named_parameters():
        assert param.grad is not None, f"ERROR - Gradient not computed for {name}: {param}"


def test_sine_encoding_odd_features():
    """Testing if SineEncoding raises an error for odd feature numbers."""
    with pytest.raises(AssertionError, match="The number of features must be even"):
        SineEncoding(features=15)