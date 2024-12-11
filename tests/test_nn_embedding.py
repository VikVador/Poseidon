r"""Tests for the poseidon.network.embedding module."""

import pytest
import random
import torch

# isort: split
from poseidon.network.embedding import SineLayer, SirenEmbedding

# Generating random dimensions for testing
BATCH, IN_FEATURES, OUT_FEATURES, N_LAYERS = (random.randint(1, 16) for _ in range(4))
IN_FEATURES *= 2  # SineEncoding requires an even number of features
OMEGA_0 = 30.0


@pytest.fixture
def sine_layer():
    """Initialize a SineLayer instance."""
    return SineLayer(in_features=IN_FEATURES, out_features=OUT_FEATURES, omega_0=OMEGA_0)


@pytest.fixture
def fake_input():
    """Fixture to provide input tensor for testing."""
    return torch.randn(BATCH, IN_FEATURES)


def test_sine_layer_output_shape(sine_layer, fake_input):
    """Testing the output shape of the SineLayer."""
    output = sine_layer(fake_input)
    expected_shape = (BATCH, OUT_FEATURES)
    assert (
        output.shape == expected_shape
    ), f"ERROR - Output shape {output.shape} does not match expected shape {expected_shape}."


def test_sine_layer_forward_consistency(sine_layer, fake_input):
    """Testing the forward pass consistency."""
    output1 = sine_layer(fake_input)
    output2 = sine_layer(fake_input)
    assert output1.shape == output2.shape, "ERROR - Inconsistent output shapes."
    assert torch.allclose(
        output1, output2
    ), "ERROR - Forward pass is not consistent for the same input."


def test_sine_layer_differentiability(sine_layer, fake_input):
    """Testing if the SineLayer is differentiable."""
    output = sine_layer(fake_input)
    loss = output.sum()
    loss.backward()
    for name, param in sine_layer.named_parameters():
        assert param.grad is not None, f"ERROR - Gradient not computed for {name} : {param}"


def test_sine_layer_invalid_input_shape(sine_layer):
    """Testing if the SineLayer raises an error for invalid input shapes."""
    invalid_input = torch.randn(BATCH, IN_FEATURES + 1)
    with pytest.raises(RuntimeError, match="mat1 and mat2 shapes cannot be multiplied"):
        sine_layer(invalid_input)


@pytest.fixture
def siren_embedding():
    """Initialize a SirenEmbedding instance."""
    return SirenEmbedding(in_features=IN_FEATURES, out_features=OUT_FEATURES, n_layers=N_LAYERS)


def test_siren_embedding_output_shape(siren_embedding, fake_input):
    """Testing SirenEmbedding output shape."""
    output = siren_embedding(fake_input)
    assert (
        output.shape == (BATCH, OUT_FEATURES)
    ), f"ERROR - Output shape {output.shape} does not match expected shape {(BATCH, OUT_FEATURES)}."


def test_siren_embedding_forward_consistency(siren_embedding, fake_input):
    """Testing the forward pass consistency."""
    output1 = siren_embedding(fake_input)
    output2 = siren_embedding(fake_input)
    assert output1.shape == output2.shape, "ERROR - Inconsistent output shapes."
    assert torch.allclose(
        output1, output2
    ), "ERROR - Forward pass is not consistent for the same input."


def test_siren_embedding_differentiability(siren_embedding, fake_input):
    """Testing if the SirenEmbedding is differentiable."""
    output = siren_embedding(fake_input)
    loss = output.sum()
    loss.backward()
    for name, param in siren_embedding.named_parameters():
        assert param.grad is not None, f"ERROR - Gradient not computed for {name} : {param}"


def test_siren_embedding_invalid_input_shape(siren_embedding):
    """Testing if the SirenEmbedding raises an error for invalid input shapes."""
    invalid_input = torch.randn(BATCH, IN_FEATURES * 2)
    with pytest.raises(RuntimeError, match="shapes cannot be multiplied"):
        siren_embedding(invalid_input)
