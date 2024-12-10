r"""Tests for the poseidon.network.residual module."""

import pytest
import random
import torch

# isort: split
from poseidon.network.residual import (
    ModulatedResidualBlock,
    SpatialModulatedResidualBlock,
    TemporalModulatedResidualBlock,
)

# Generating random dimensions for testing
BATCH, CHANNELS, MOD_FEATURES, TIME, HEIGHT, WIDTH = (random.randint(1, 16) for _ in range(6))


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
def modulated_block():
    """Fixture to provide an instance of ModulatedResidualBlock."""
    return ModulatedResidualBlock(
        channels=CHANNELS,
        mod_features=MOD_FEATURES,
        spatial=2,
        kernel_size=3,
        stride=1,
        padding=1,
    )


@pytest.fixture
def spatial_modulated_block():
    """Fixture to provide an instance of SpatialModulatedResidualBlock."""
    return SpatialModulatedResidualBlock(
        channels=CHANNELS,
        mod_features=MOD_FEATURES,
        kernel_size=3,
        stride=1,
        padding=1,
    )


@pytest.fixture
def temporal_modulated_block():
    """Fixture to provide an instance of TemporalModulatedResidualBlock."""
    return TemporalModulatedResidualBlock(
        channels=CHANNELS,
        mod_features=MOD_FEATURES,
        kernel_size=3,
        stride=1,
        padding=1,
    )


def test_modulated_residual_block_forward(modulated_block, fake_input, fake_modulation):
    """Testing that ModulatedResidualBlock runs a forward pass successfully."""
    output = modulated_block(fake_input, fake_modulation)
    assert output.shape == fake_input.shape, f"ERROR - Output shape {output.shape} is incorrect"


def test_spatial_modulated_residual_block_forward(
    spatial_modulated_block, fake_input, fake_modulation
):
    """Testing that SpatialModulatedResidualBlock runs a forward pass successfully."""
    output = spatial_modulated_block(fake_input, fake_modulation)
    assert output.shape == fake_input.shape, f"ERROR - Output shape {output.shape} is incorrect"


def test_temporal_modulated_residual_block_forward(
    temporal_modulated_block, fake_input, fake_modulation
):
    """Testing that TemporalModulatedResidualBlock runs a forward pass successfully."""
    output = temporal_modulated_block(fake_input, fake_modulation)
    assert output.shape == fake_input.shape, f"ERROR - Output shape {output.shape} is incorrect"


def test_modulated_block_with_invalid_input_shape(modulated_block):
    """Testing that ModulatedResidualBlock raises an error for an invalid input shape."""
    invalid_input = torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH)
    invalid_modulation = torch.randn(BATCH, MOD_FEATURES)
    with pytest.raises(ValueError, match="not enough values to unpack"):
        modulated_block(invalid_input, invalid_modulation)


def test_spatial_modulated_block_with_invalid_input_shape(spatial_modulated_block):
    """Testing that SpatialModulatedResidualBlock raises an error for an invalid input shape."""
    invalid_input = torch.randn(BATCH, CHANNELS, TIME, HEIGHT)
    invalid_modulation = torch.randn(BATCH, MOD_FEATURES)
    with pytest.raises(ValueError, match="not enough values to unpack"):
        spatial_modulated_block(invalid_input, invalid_modulation)


def test_temporal_modulated_block_with_invalid_input_shape(temporal_modulated_block):
    """Testing that TemporalModulatedResidualBlock raises an error for an invalid input shape."""
    invalid_input = torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH)
    invalid_modulation = torch.randn(BATCH, MOD_FEATURES)
    with pytest.raises(ValueError, match="not enough values to unpack"):
        temporal_modulated_block(invalid_input, invalid_modulation)


def test_modulated_residual_block_differentiability(
    spatial_modulated_block, fake_input, fake_modulation
):
    """Testing if the Modulated Residual Block is differentiable."""
    output = spatial_modulated_block(fake_input, fake_modulation)
    loss = output.sum()
    loss.backward()
    for name, param in spatial_modulated_block.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"ERROR - Gradient not computed for {name} : {param}"
        else:
            assert (
                param.grad is None
            ), f"ERROR - Unexpected gradient for non-learnable parameter: {name}"


def test_temporal_modulated_residual_block_differentiability(
    temporal_modulated_block, fake_input, fake_modulation
):
    """Testing if the Temporal Modulated Residual Block is differentiable."""
    output = temporal_modulated_block(fake_input, fake_modulation)
    loss = output.sum()
    loss.backward()

    # Check gradients for each parameter
    for name, param in temporal_modulated_block.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"ERROR - Gradient not computed for {name} : {param}"
        else:
            assert (
                param.grad is None
            ), f"ERROR - Unexpected gradient for non-learnable parameter: {name}"


def test_forward_with_dropout(modulated_block, fake_input, fake_modulation):
    """Testing that ModulatedResidualBlock with dropout works."""
    modulated_block_with_dropout = ModulatedResidualBlock(
        channels=CHANNELS,
        mod_features=MOD_FEATURES,
        spatial=2,
        kernel_size=3,
        stride=1,
        padding=1,
        dropout=0.5,
    )
    output = modulated_block(fake_input, fake_modulation)
    output_with_dropout = modulated_block_with_dropout(fake_input, fake_modulation)
    assert output.shape == fake_input.shape, f"ERROR - Output shape {output.shape} is incorrect"
    assert (
        output_with_dropout.shape == fake_input.shape
    ), f"ERROR - Output shape {output_with_dropout.shape} is incorrect"
