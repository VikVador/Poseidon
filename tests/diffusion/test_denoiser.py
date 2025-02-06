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
MESH_LEVELS, (MESH_LAT, MESH_LON) = (
    random.randint(3, 5),
    (random.randint(24, 32) for _ in range(2)),
)

(INPUT_B, INPUT_C, INPUT_K), INPUT_H, INPUT_W = (
    (random.choice([3, 5]) for _ in range(3)),
    10,
    10,
)

UNET_KERNEL, UNET_FEATURES, UNET_CHANNELS, UNET_BLOCKS, SIREN_FEATURES, SIREN_LAYERS = (
    random.choice([3, 5]),
    random.choice([1]),
    list(random.randint(2, 5) for _ in range(3)),
    list(random.choice([1, 2]) for _ in range(3)),
    random.choice([2, 4]),
    random.choice([2, 3]),
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
    sigma = torch.randn(INPUT_B, UNET_FEATURES)
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
        "hid_channels": UNET_CHANNELS,
        "hid_blocks": UNET_BLOCKS,
        "kernel_size": UNET_KERNEL,
        "mod_features": UNET_FEATURES,
    }
    config_siren = {
        "features": SIREN_FEATURES,
        "n_layers": SIREN_LAYERS,
    }
    config_region = {
        "latitude": slice(0, 10),
        "longitude": slice(0, 10),
    }
    dimensions = (INPUT_B, INPUT_C, INPUT_K, INPUT_H, INPUT_W)
    return dimensions, config_unet, config_siren, config_region


@pytest.fixture
def backbone(fake_zarr_mesh, fake_configurations):
    """Initialize a PoseidonBackbone instance."""
    dimensions, config_unet, config_siren, config_region = fake_configurations

    return PoseidonBackbone(
        dimensions=dimensions,
        config_unet=config_unet,
        config_siren=config_siren,
        config_region=config_region,
        path_mesh=fake_zarr_mesh,
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
