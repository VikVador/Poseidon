r"""Tests for the poseidon.training.load module."""

import numpy as np
import pytest
import random
import torch
import xarray as xr

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

# isort: split
from poseidon.data.const import TOY_DATASET_REGION
from poseidon.diffusion.backbone import PoseidonBackbone
from poseidon.training.load import load_backbone, load_optimizer, load_scheduler
from poseidon.training.save import PoseidonSave

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
    """Initialize an optimizer instance."""
    optimizer = Adam(fake_backbone.parameters(), lr=0.01)
    optimizer.param_groups[0]["lr"] = 0.02  # Simulate step
    return optimizer


@pytest.fixture
def fake_scheduler(fake_optimizer):
    """Initialize an scheduler instance."""
    scheduler = StepLR(fake_optimizer, step_size=1, gamma=0.9)
    scheduler.step()
    return scheduler


@pytest.fixture
def save_optimizer_state(fake_optimizer, tmp_path):
    """Save the state of an optimizer."""
    path = tmp_path / "test_model" / "tools"
    path.mkdir(parents=True, exist_ok=True)
    torch.save({"optimizer_state_dict": fake_optimizer.state_dict()}, path / "optimizer.pth")
    return tmp_path


@pytest.fixture
def save_scheduler_state(fake_scheduler, tmp_path):
    """Save the state of an scheduler."""
    path = tmp_path / "test_model" / "tools"
    path.mkdir(parents=True, exist_ok=True)
    torch.save({"scheduler_state_dict": fake_scheduler.state_dict()}, path / "scheduler.pth")
    return tmp_path


def test_load_optimizer(fake_backbone, save_optimizer_state):
    """Testing if optimizer is properly loaded from checkpoint."""

    new_optimizer = Adam(fake_backbone.parameters(), lr=0.01)
    initial_state = new_optimizer.state_dict()
    load_optimizer("test_model", new_optimizer, path=save_optimizer_state)
    assert (
        new_optimizer.state_dict() != initial_state
    ), "ERROR - Optimizer state was not updated after loading."


def test_load_scheduler(fake_backbone, save_scheduler_state):
    """Testing if scheduler is properly loaded from checkpoint."""

    new_optimizer = Adam(fake_backbone.parameters(), lr=0.01)
    new_scheduler = StepLR(new_optimizer, step_size=1, gamma=0.9)
    initial_state = new_scheduler.state_dict()
    load_scheduler("test_model", new_scheduler, path=save_scheduler_state)
    assert (
        new_scheduler.state_dict() != initial_state
    ), "ERROR - Scheduler state was not updated after loading."


def test_load_backbone(temp_dir, fake_backbone, fake_zarr_mesh, fake_configurations):
    """Testing if a PoseidonBackbone is save and loaded correctly."""

    # Initialization
    name_model, (dimensions, config_unet, config_siren, _, config_transformer) = (
        "test_model",
        fake_configurations,
    )

    # Saving the model with custom tool
    poseidon_save = PoseidonSave(
        path=temp_dir,
        name_model=name_model,
        variables=["votemper"],
        dimensions=dimensions,
        config_unet=config_unet,
        config_transformer=config_transformer,
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

    assert isinstance(
        loaded_model, PoseidonBackbone
    ), "ERROR - Loaded model is not a PoseidonBackbone."
    assert all(
        torch.equal(p1, p2)
        for p1, p2 in zip(loaded_model.parameters(), fake_backbone.parameters())
    ), "ERROR - Loaded model parameters do not match the saved model parameters."

    # Ensure model is in training mode by default
    assert loaded_model.training, "ERROR - Loaded model is not in training mode."
