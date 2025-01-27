r"""Tests for the poseidon.training.load module."""

import numpy as np
import pytest
import random
import torch
import xarray as xr

# isort: split
from poseidon.diffusion.backbone import PoseidonBackbone
from poseidon.training.load import load_backbone
from poseidon.training.save import PoseidonSave

# Generating random dimensions for testing
MESH_LEVELS, MESH_LAT, MESH_LON = (4, 32, 32)

(INPUT_B, INPUT_C, INPUT_K), INPUT_H, INPUT_W = (
    (random.randint(3, 5) for _ in range(3)),
    10,
    10,
)

UNET_KERNEL, UNET_FEATURES, UNET_CHANNELS, UNET_BLOCKS, SIREN_FEATURES, SIREN_LAYERS = (
    random.choice([3, 5]),
    random.choice([3, 4]),
    list(random.randint(2, 5) for _ in range(3)),
    list(random.choice([1, 2]) for _ in range(3)),
    random.choice([2, 4]),
    random.choice([2, 3]),
)


@pytest.fixture
def temp_dir(tmp_path):
    """Fixture to provide a temporary directory for testing."""
    return tmp_path


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

    dimensions = [
        INPUT_B,
        INPUT_C,
        INPUT_K,
        INPUT_H,
        INPUT_W,
    ]

    config_unet = {
        "hid_channels": UNET_CHANNELS,
        "hid_blocks": UNET_BLOCKS,
        "kernel_size": UNET_KERNEL,
        "mod_features": UNET_FEATURES,
    }

    config_siren = {
        "features": 2,
        "n_layers": 1,
    }

    config_region = {
        "latitude": slice(0, MESH_LAT),
        "longitude": slice(0, MESH_LON),
        "level": slice(0, MESH_LEVELS),
    }

    return dimensions, config_unet, config_siren, config_region


@pytest.fixture
def fake_backbone(fake_zarr_mesh, fake_configurations):
    """Initialize a PoseidonBackbone instance."""
    dimensions, config_unet, config_siren, config_region = fake_configurations

    return PoseidonBackbone(
        dimensions=dimensions,
        config_unet=config_unet,
        config_siren=config_siren,
        config_region=config_region,
        path_mesh=fake_zarr_mesh,
    )


def test_load_backbone(temp_dir, fake_backbone, fake_configurations):
    """Testing if a PoseidonBackbone is save and loaded correctly."""

    # Initialization
    name_model, (dimensions, config_unet, config_siren, _) = ("test_model", fake_configurations)

    # Saving the model with custom tool
    poseidon_save = PoseidonSave(
        path=temp_dir,
        name_model=name_model,
        dimensions=dimensions,
        config_unet=config_unet,
        config_siren=config_siren,
        config_problem={"toy_problem": True},
    )

    poseidon_save.save(
        loss=0.0,
        model=fake_backbone,
    )

    # Loading the model with custom tool
    loaded_model = load_backbone(
        name_model=name_model,
        path=temp_dir,
        best=True,
        backup=False,
    )

    # Assertions
    assert isinstance(
        loaded_model, PoseidonBackbone
    ), "ERROR - Loaded model is not a PoseidonBackbone."
    assert all(
        torch.equal(p1, p2)
        for p1, p2 in zip(loaded_model.parameters(), fake_backbone.parameters())
    ), "ERROR - Loaded model parameters do not match the saved model parameters."

    # Ensure model is in training mode by default
    assert loaded_model.training, "ERROR - Loaded model is not in training mode."
