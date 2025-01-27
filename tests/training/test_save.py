r"""Tests for the poseidon.training.save module."""

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

# isort: split
from poseidon.training.save import (
    save_backbone,
    save_configuration,
    save_tools,
)

# Generating random dimensions for testing
INPUT_SIZE, OUTPUT_SIZE = (np.random.randint(1, 10) for _ in range(2))


# fmt: off
@pytest.fixture
def fake_backbone():
    """A fake Backbone model for testing purposes."""
    return nn.Linear(INPUT_SIZE, OUTPUT_SIZE)

@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


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
