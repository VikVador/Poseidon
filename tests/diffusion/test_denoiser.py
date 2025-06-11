r"""Tests for the poseidon.diffusion.denoiser module."""

import numpy as np
import pytest
import random
import torch
import xarray as xr

from torch import nn

# isort: split
from poseidon.diffusion.backbone import PoseidonBackbone
from poseidon.diffusion.denoiser import PoseidonDenoiser

# Generating random dimensions for testing
MESH_LEVELS, MESH_LAT, MESH_LON = (
    8,
    32,
    32,
)

INPUT_B, INPUT_C, INPUT_K, INPUT_H, INPUT_W = (
    random.choice([3, 5]),
    4,
    random.choice([3, 5]),
    16,
    16,
)

UNET_KERNEL, UNET_FEATURES, UNET_SCALING, UNET_BLOCKS, UNET_CHANNELS = (
    random.choice([3, 5]),
    random.choice([2, 4]),
    random.choice([1, 2]),
    list(random.choice([1, 2]) for _ in range(3)),
    list(random.randint(2, 5) for _ in range(3)),
)

TRANSF_CHANNELS, TRANSF_BLOCKS, TRANSF_PATCH, TRANSF_SCALING, TRANSF_HEADS = (
    random.choice([8, 32]),
    random.choice([4, 8]),
    random.choice([1, 2]),
    random.choice([1, 2]),
    random.choice([1, 2]),
)

SIREN_FEATURES, SIREN_LAYERS = (
    random.choice([2, 4]),
    random.choice([1, 2]),
)


@pytest.fixture
def fake_input():
    """Fixture to provide a noisy input tensor for testing."""
    x = torch.randn(INPUT_B, (INPUT_C * INPUT_K * INPUT_H * INPUT_W))
    x.requires_grad = True
    x = x.to(dtype=torch.float32)
    return x


@pytest.fixture
def fake_noise():
    """Fixture to provide a noise level tensor for testing."""
    sigma = torch.randn(INPUT_B, 1)
    sigma.requires_grad = True
    sigma = sigma.to(dtype=torch.float32)
    return sigma


@pytest.fixture
def fake_zarr_mesh(tmp_path):
    """Creates a fake Zarr mesh dataset for testing."""

    ds = xr.Dataset(
        {
            name: (dims, np.random.rand(MESH_LEVELS, MESH_LAT, MESH_LON))
            for name, dims in {
                "x_mesh": ("level", "latitude", "longitude"),
                "y_mesh": ("level", "latitude", "longitude"),
                "z_mesh": ("level", "latitude", "longitude"),
            }.items()
        },
    )
    path = tmp_path / "fake_mesh.zarr"
    ds.to_zarr(path)
    return path


@pytest.fixture
def fake_configurations():
    """Provides random configurations for UNet, Siren, and the spatial region."""
    config_unet = {
        "kernel_size": UNET_KERNEL,
        "mod_features": UNET_FEATURES,
        "ffn_scaling": UNET_SCALING,
        "hid_blocks": UNET_BLOCKS,
        "hid_channels": UNET_CHANNELS,
        "attention_heads": {"-1": 1},
    }
    config_transformer = {
        "hid_channels": TRANSF_CHANNELS,
        "hid_blocks": TRANSF_BLOCKS,
        "patch_size": TRANSF_PATCH,
        "ffn_scaling": TRANSF_SCALING,
        "attention_heads": TRANSF_HEADS,
    }
    config_siren = {
        "features": SIREN_FEATURES,
        "n_layers": SIREN_LAYERS,
    }
    config_region = {
        "latitude": slice(0, INPUT_H),
        "longitude": slice(0, INPUT_W),
        "level": slice(0, INPUT_C),
    }
    dimensions = (INPUT_B, INPUT_C, INPUT_K, INPUT_H, INPUT_W)
    return dimensions, config_unet, config_siren, config_region, config_transformer


@pytest.fixture
def backbone(fake_zarr_mesh, fake_configurations):
    """Initialize a PoseidonBackbone instance."""
    dimensions, config_unet, config_siren, config_region, config_transformer = fake_configurations

    return PoseidonBackbone(
        variables=["votemper"],
        dimensions=dimensions,
        config_unet=config_unet,
        config_siren=config_siren,
        config_region=config_region,
        config_transformer=config_transformer,
    )


@pytest.fixture
def denoiser(backbone):
    """Fixture to create an instance of PoseidonDenoiser."""
    return PoseidonDenoiser(backbone=backbone)


def test_denoiser_initialization(backbone):
    """Testing the initialization."""
    denoiser = PoseidonDenoiser(backbone)
    assert isinstance(
        denoiser, nn.Module
    ), "ERROR - PoseidonDenoiser should inherit from torch.nn.Module."
    assert hasattr(
        denoiser, "backbone"
    ), "ERROR - Denoiser should possess a 'backbone' attribute (PoseidonBackbone)."
    assert (
        denoiser.backbone == backbone
    ), "ERROR - Backbone instance is not the same as the one given in 'init'."


def testing_denoiser_forward_consistency(denoiser, fake_input, fake_noise):
    """Testing the forward pass consistency."""
    output1 = denoiser.forward(fake_input, fake_noise)
    output2 = denoiser.forward(fake_input, fake_noise)
    assert output1.shape == output2.shape, "ERROR - Inconsistent output shapes."
    assert torch.allclose(
        output1,
        output2,
        equal_nan=True,
    ), "ERROR - Forward pass is not consistent for the same input."


def test_denoiser_differentiability(denoiser, fake_input, fake_noise):
    """Testing that the PoseidonDenoiser is differentiable."""
    output = denoiser(fake_input, fake_noise)
    loss = output.sum()
    loss.backward()
    for name, param in denoiser.named_parameters():
        assert param.grad is not None, f"ERROR - Gradient not computed for {name} : {param}"


def test_denoiser_invalid_input_shape(backbone):
    """Testing if the PoseidonDenoiser raises an error for invalid input shapes."""
    invalid_x = torch.randn(5, 5, 3)
    sigma = torch.rand(5, 1)
    with pytest.raises(RuntimeError, match="while processing rearrange-reduction"):
        backbone.forward(invalid_x, sigma)
