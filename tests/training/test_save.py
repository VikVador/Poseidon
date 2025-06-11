r"""Tests for the poseidon.training.save module."""

import numpy as np
import pytest
import random
import torch
import xarray as xr
import yaml

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

# isort: split
from poseidon.data.const import TOY_DATASET_REGION
from poseidon.diffusion.backbone import PoseidonBackbone
from poseidon.training.load import load_backbone, load_optimizer, load_scheduler
from poseidon.training.save import PoseidonSave, save_backbone, save_configuration, save_tools

# Generating random dimensions for testing
MESH_LEVELS, MESH_LAT, MESH_LON = (
    1,
    128,
    256,
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
def fake_backbone(fake_zarr_mesh, fake_configurations):
    """Initialize a PoseidonBackbone instance."""
    dimensions, config_unet, config_siren, config_region, config_transformer = fake_configurations
    return PoseidonBackbone(
        variables=["votemper"],
        dimensions=dimensions,
        config_unet=config_unet,
        config_siren=config_siren,
        config_region=TOY_DATASET_REGION,
        config_transformer=config_transformer,
    )


@pytest.fixture
def fake_optimizer(fake_backbone):
    return Adam(fake_backbone.parameters(), lr=0.001)


@pytest.fixture
def fake_scheduler(fake_optimizer):
    return StepLR(fake_optimizer, step_size=10, gamma=0.1)


def test_save_backbone(temp_dir, fake_backbone):
    """Testing if a model is saved correctly."""

    path, name_model, name_state = (
        temp_dir,
        "test_model",
        "test_state",
    )

    save_backbone(
        path=path,
        model=fake_backbone,
        name_model=name_model,
        name_state=name_state,
    )

    model_folder = path / name_model / "models"
    assert model_folder.exists(), "ERROR - Model folder was not created."
    assert (
        model_folder / f"{name_state}.pth"
    ).exists(), "ERROR - Model state file (.pth) was not saved."


def test_save_configuration(temp_dir):
    """Testing if a configuration is saved correctly."""

    path, name_model, name_config, config = (
        temp_dir,
        "test_model",
        "test_config",
        {
            "alpha": 10,
            "beta": 20,
        },
    )

    save_configuration(
        path=path,
        config=config,
        name_model=name_model,
        name_config=name_config,
    )

    config_folder = path / name_model / "configurations"
    config_path = config_folder / f"{name_config}.yml"
    with open(config_path, "r") as file:
        saved_config = yaml.safe_load(file)

    assert config_folder.exists(), "ERROR - Configuration folder was not created."
    assert config_path.exists(),   "ERROR - Configuration file was not saved."
    assert saved_config == config, "ERROR - Saved configuration does not match the original."


def test_save_tools(temp_dir, fake_optimizer, fake_scheduler):
    """Testing if the optimizer and scheduler are saved correctly."""

    path, name_model = (
        temp_dir,
        "test_model",
    )

    save_tools(
        path=path,
        name_model=name_model,
        optimizer=fake_optimizer,
        scheduler=fake_scheduler,
    )

    tools_folder = path / name_model / "tools"
    optimizer_path = tools_folder / "optimizer.pth"
    scheduler_path = tools_folder / "scheduler.pth"

    assert tools_folder.exists(),   "ERROR - Tools folder was not created."
    assert optimizer_path.exists(), "ERROR - Optimizer state file was not saved."
    assert scheduler_path.exists(), "ERROR - Scheduler state file was not saved."

    # Verify optimizer state
    loaded_optimizer_state = torch.load(optimizer_path, weights_only=True)["optimizer_state_dict"]
    assert (
        loaded_optimizer_state == fake_optimizer.state_dict()
    ), "ERROR - Optimizer state does not match."

    # Verify scheduler state
    loaded_scheduler_state = torch.load(scheduler_path, weights_only=True)["scheduler_state_dict"]
    assert (
        loaded_scheduler_state == fake_scheduler.state_dict()
    ), "ERROR - Scheduler state does not match."
