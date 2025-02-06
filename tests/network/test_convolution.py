r"""Tests for the poseidon.network.convolution module."""

import pytest
import random
import torch

# isort: split
from poseidon.network.convolution import ConvNd

# Generating random dimensions for testing
BATCH, CHANNELS, TIME, HEIGHT, WIDTH = (random.randint(1, 16) for _ in range(5))


@pytest.fixture
def fake_input():
    """Fixture to provide a fake input tensor for testing."""
    return torch.randn(BATCH, CHANNELS, TIME, HEIGHT, WIDTH)


def test_convnd_valid_conv_2d():
    """Testing ConvNd to ensure it returns a valid 2D convolutional layer."""
    conv_layer = ConvNd(
        in_channels=CHANNELS, out_channels=CHANNELS, spatial=2, kernel_size=3, stride=1, padding=1
    )
    assert isinstance(conv_layer, torch.nn.Conv2d), "ERROR - ConvNd did not return a Conv2d layer."


def test_convnd_invalid_spatial():
    """Testing ConvNd to ensure it raises an error when an invalid spatial dimension is provided."""
    with pytest.raises(NotImplementedError):
        ConvNd(in_channels=CHANNELS, out_channels=CHANNELS, spatial=4, kernel_size=3)
