r"""Tests for UNet block in poseidon.network.unet module."""

import pytest
import random
import torch

# isort: split
from poseidon.network.unet import UNetBlock

# Generating random dimensions for testing
BATCH, CHANNELS, MOD_FEATURES, TIME, HEIGHT, WIDTH = (random.randint(2, 16) for _ in range(6))


@pytest.fixture
def fake_input():
    """Fixture to provide a fake input tensor."""
    x = torch.randn(BATCH, CHANNELS, TIME, HEIGHT, WIDTH)
    x.requires_grad = True
    return x


@pytest.fixture
def fake_modulation():
    """Fixture to provide a fake modulation vector."""
    mod = torch.randn(BATCH, MOD_FEATURES)
    mod.requires_grad = True
    return mod


@pytest.fixture
def unet_block():
    """Fixture to provide an instance of UNetBlock."""
    return UNetBlock(
        channels=CHANNELS,
        mod_features=MOD_FEATURES,
        dropout=0.5,
        kernel_size=3,
        padding=1,
        stride=1,
    )


def test_unet_block_forward(unet_block, fake_input, fake_modulation):
    """Testing that the forward pass of UNetBlock works as expected."""
    output = unet_block(fake_input, fake_modulation)
    assert output.shape == fake_input.shape, f"ERROR - Output shape {output.shape} is incorrect"


def test_unet_block_differentiability(unet_block, fake_input, fake_modulation):
    """Testing if the UNetBlock is differentiable."""
    output = unet_block(fake_input, fake_modulation)
    loss = output.sum()
    loss.backward()

    for name, param in unet_block.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"ERROR - Gradient not computed for {name} : {param}"
        else:
            assert (
                param.grad is None
            ), f"ERROR - Unexpected gradient for non-learnable parameter: {name}"


def test_unet_block_invalid_input_shape(unet_block):
    """Testing that UNetBlock raises an error for invalid input shapes."""
    invalid_input = torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH)
    invalid_modulation = torch.randn(BATCH, MOD_FEATURES)
    with pytest.raises(ValueError, match="not enough values to unpack"):
        unet_block(invalid_input, invalid_modulation)


def test_unet_block_dropout_behavior(unet_block, fake_input, fake_modulation):
    """Testing UNetBlock behavior with different dropout values."""
    unet_block_with_dropout = UNetBlock(
        channels=CHANNELS,
        mod_features=MOD_FEATURES,
        dropout=0.5,
        kernel_size=3,
        padding=1,
    )
    unet_block_no_dropout = UNetBlock(
        channels=CHANNELS,
        mod_features=MOD_FEATURES,
        dropout=0.0,
        kernel_size=3,
        padding=1,
    )

    output_with_dropout = unet_block_with_dropout(fake_input, fake_modulation)
    output_no_dropout = unet_block_no_dropout(fake_input, fake_modulation)
    output_original = unet_block(fake_input, fake_modulation)

    assert (
        output_original.shape == fake_input.shape
    ), f"ERROR - Output shape {output_original.shape} is incorrect"
    assert (
        output_with_dropout.shape == fake_input.shape
    ), f"ERROR - Output shape {output_with_dropout.shape} is incorrect"
    assert (
        output_no_dropout.shape == output_with_dropout.shape
    ), "ERROR - Output shape mismatch between dropout variants."
    assert not torch.allclose(
        output_no_dropout, output_with_dropout
    ), "ERROR - Dropout is not applied as expected."
