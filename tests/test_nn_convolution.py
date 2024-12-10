r"""Tests for the poseidon.network.convolution module."""

import pytest
import random
import torch

# isort: split
from poseidon.network.convolution import ConvNd, Convolution2DBlock

# Generating random dimensions for testing
BATCH, CHANNELS, TIME, HEIGHT, WIDTH = (random.randint(1, 16) for _ in range(5))


@pytest.fixture
def conv2d_block():
    """Initialize a Convolution2DBlock instance."""
    return Convolution2DBlock(
        in_channels=CHANNELS, out_channels=CHANNELS, kernel_size=3, stride=1, padding=1
    )


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


def test_conv2d_block_consistency(conv2d_block, fake_input):
    """Testing that the forward pass produces consistent results for the same input."""
    output1 = conv2d_block(fake_input)
    output2 = conv2d_block(fake_input)
    assert output1.shape == output2.shape, "ERROR - Inconsistent output shapes."
    assert torch.allclose(
        output1, output2
    ), "ERROR - Forward pass is not consistent for the same input."


def test_conv2d_block_output_shape(conv2d_block, fake_input):
    """Testing that Convolution2DBlock correctly processes input and outputs the expected shape."""
    output = conv2d_block(fake_input)
    expected_height = HEIGHT
    expected_width = WIDTH
    assert output.shape == (BATCH, CHANNELS, TIME, expected_height, expected_width), (
        f"ERROR - Output shape {output.shape} does not match expected shape "
        f"(BATCH, CHANNELS, TIME, {expected_height}, {expected_width}). Height and width mismatch."
    )


def test_conv2d_block_invalid_input_shape(conv2d_block):
    """Testing that Convolution2DBlock raises an error for an invalid input shape."""
    invalid_input = torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH)  # Missing TIME dimension
    with pytest.raises(ValueError, match="not enough values to unpack"):
        conv2d_block(invalid_input)


def test_conv2d_block_gradient_computation(conv2d_block, fake_input):
    """Testing that Convolution2DBlock properly computes gradients during backpropagation."""
    output = conv2d_block(fake_input)
    loss = output.sum()
    loss.backward()
    for param in conv2d_block.parameters():
        assert param.grad is not None, f"ERROR - Gradient not computed for parameter: {param}"


@pytest.mark.parametrize(
    "kernel_size, stride, padding, expected_height, expected_width",
    [
        (3, 1, 1, HEIGHT, WIDTH),
        (5, 2, 2, (HEIGHT // 2) + (HEIGHT % 2), (WIDTH // 2) + (WIDTH % 2)),
        (7, 1, 3, HEIGHT, WIDTH),
    ],
)
def test_conv2d_block_with_different_kernel_stride_padding(
    kernel_size, stride, padding, expected_height, expected_width, conv2d_block, fake_input
):
    """Testing Convolution2DBlock with different kernel sizes, strides, and paddings."""
    conv2d_block = Convolution2DBlock(
        in_channels=CHANNELS,
        out_channels=CHANNELS,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )
    output = conv2d_block(fake_input)

    assert output.shape == (BATCH, CHANNELS, TIME, expected_height, expected_width), (
        f"ERROR - Output shape {output.shape} does not match expected shape "
        f"(BATCH, CHANNELS, TIME, {expected_height}, {expected_width}) for kernel_size={kernel_size}, "
        f"stride={stride}, padding={padding}."
    )
