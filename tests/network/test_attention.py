r"""Tests for the poseidon.network.attention module."""

import pytest
import random
import torch


# iso: split
from poseidon.network.attention import SelfAttentionNd

# Generating random dimensions for testing
BATCH, CHANNELS, TIME, HEIGHT, WIDTH = (random.randint(1, 4) for _ in range(5))
CHANNELS *= 2  # Ensure that the number of channels is even


@pytest.fixture
def fake_input():
    """Fixture to provide a fake input tensor for testing."""
    return torch.randn(BATCH, CHANNELS, TIME, HEIGHT, WIDTH).requires_grad_()


def test_self_attention_initialization():
    """Testing if SelfAttentionNd initializes properly."""
    self_attention_block = SelfAttentionNd(channels=8, heads=2)
    assert (
        self_attention_block.embed_dim == 8
    ), "ERROR - Embedding dimension should match the number of channels."
    assert (
        self_attention_block.num_heads == 2
    ), "ERROR - Number of heads does not match the specified value."


def test_self_attention_output_shape(fake_input):
    """Testing if the output shape is correct."""
    self_attention_block = SelfAttentionNd(channels=CHANNELS, heads=2)
    output = self_attention_block(fake_input)
    assert (
        output.shape == fake_input.shape
    ), f"ERROR - Output shape {output.shape} does not match input shape {fake_input.shape}."


def test_self_attention_gradient(fake_input):
    """Testing that SelfAttentionNd is differentiable."""
    self_attention_block = SelfAttentionNd(channels=CHANNELS, heads=2)
    output = self_attention_block(fake_input)
    loss = output.sum()
    loss.backward()
    for name, param in self_attention_block.named_parameters():
        assert param.grad is not None, f"ERROR - Gradient not computed for {name} : {param}"
