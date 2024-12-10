r"""Tests for the poseidon.network.upsample module."""

import pytest
import random
import torch

# isort: split
from poseidon.network.upsample import UpsampleBlock

# Generating random dimensions for testing
BATCH, CHANNELS, TIME, HEIGHT, WIDTH = (random.randint(1, 16) for _ in range(5))
SCALE_FACTOR = random.randint(2, 4)


@pytest.fixture
def upsample_block():
    """Initialize an UpsampleBlock instance."""
    return UpsampleBlock(scale_factor=SCALE_FACTOR, mode="nearest")


@pytest.fixture
def fake_input():
    """Fixture to provide a 5D input tensor for testing."""
    x = torch.randn(BATCH, CHANNELS, TIME, HEIGHT, WIDTH)
    x.requires_grad = True
    return x


def test_upsample_block_output_shape(upsample_block, fake_input):
    """Testing upsampling block output shape."""
    output = upsample_block(fake_input)
    expected_shape = (
        BATCH,
        CHANNELS,
        TIME,
        HEIGHT * SCALE_FACTOR,
        WIDTH * SCALE_FACTOR,
    )
    assert (
        output.shape == expected_shape
    ), f"ERROR - Output shape {output.shape} does not match expected shape {expected_shape}."


def test_upsample_block_forward_consistency(upsample_block, fake_input):
    """Testing the forward pass consistency."""
    output1 = upsample_block(fake_input)
    output2 = upsample_block(fake_input)
    assert output1.shape == output2.shape, "ERROR - Inconsistent output shapes."
    assert torch.allclose(
        output1, output2
    ), "ERROR - Forward pass results are inconsistent for the same input."


def test_upsample_block_differentiability(upsample_block, fake_input):
    """Testing if the UpsampleBlock is differentiable."""
    output = upsample_block(fake_input)
    loss = output.sum()
    loss.backward()
    for name, param in upsample_block.named_parameters():
        assert param.grad is not None, f"ERROR - Gradient not computed for {name} : {param}"


def test_upsample_block_invalid_input_shape(upsample_block):
    """Testing if the UpsampleBlock raises an error for invalid input shapes."""
    invalid_input = torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH)
    with pytest.raises(ValueError, match="not enough values to unpack"):
        upsample_block(invalid_input)


def test_upsample_block_custom_kwargs(fake_input):
    """Testing custom arguments and behavior of nn.Upsample."""
    block_nearest = UpsampleBlock(
        scale_factor=SCALE_FACTOR,
        mode="nearest",
    )
    block_bilinear = UpsampleBlock(
        scale_factor=SCALE_FACTOR,
        mode="bilinear",
        align_corners=True,
    )

    output_nearest, output_bilinear = (block_nearest(fake_input), block_bilinear(fake_input))
    expected_height, expected_width = HEIGHT * SCALE_FACTOR, WIDTH * SCALE_FACTOR

    # Computing gradient differences for nearest and bilinear upsampling
    nearest_grad = torch.mean(
        torch.abs(output_nearest[:, :, :, 1:, :] - output_nearest[:, :, :, :-1, :])
    )
    bilinear_grad = torch.mean(
        torch.abs(output_bilinear[:, :, :, 1:, :] - output_bilinear[:, :, :, :-1, :])
    )

    assert output_nearest.shape[-2:] == (
        expected_height,
        expected_width,
    ), "ERROR - Nearest neighbor upsampling dimensions are incorrect."

    assert output_bilinear.shape[-2:] == (
        expected_height,
        expected_width,
    ), "ERROR - Bilinear upsampling dimensions are incorrect."

    assert not torch.allclose(
        output_nearest, output_bilinear
    ), "ERROR - Output tensors are identical for different upsampling modes."

    assert (
        bilinear_grad < nearest_grad
    ), "ERROR - Bilinear upsampling did not produce smoother transitions than nearest neighbor."
