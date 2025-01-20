r"""Tests for the poseidon.training.save module."""

import numpy as np
import pytest

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

# isort: split
from poseidon.data.const import DATASET_REGION
from poseidon.diffusion.backbone import PoseidonBackbone


@pytest.fixture
def fake_backbone():
    config_unet = {
        "kernel_size": 3,
        "mod_features": 1,
        "hid_channels": np.random.randint(1, 4, size=4).tolist(),
        "hid_blocks": np.random.randint(1, 2, size=4).tolist(),
        "dropout": None,
        "attention_heads": {str(i): int(np.random.choice([2, 4])) for i in range(2, 4)},
    }

    config_siren = {
        "features": np.random.randint(8, 32),
        "n_layers": np.random.randint(1, 2),
    }

    return PoseidonBackbone(
        dimensions=(2, 5, 3, 32, 64),
        config_unet=config_unet,
        config_siren=config_siren,
        config_region=DATASET_REGION,
    )


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def fake_optimizer(fake_backbone):
    return Adam(fake_backbone.parameters(), lr=0.001)


@pytest.fixture
def fake_scheduler(fake_optimizer):
    return StepLR(fake_optimizer, step_size=10, gamma=0.1)


# def test_save_backbone(temp_dir, fake_backbone):
#     path = temp_dir
#     name_model = "test_model"
#     name_state = "test_state"

#     save_backbone(path, fake_backbone, name_model, name_state)

#     model_folder = path / name_model / "models"
#     assert model_folder.exists(), "Model folder was not created."
#     assert (model_folder / f"{name_state}.pth").exists(), "Model state file was not saved."


# def test_save_configuration(temp_dir):
#     path = temp_dir
#     name_model = "test_model"
#     name_config = "test_config"
#     config = {"param1": 10, "param2": 20}

#     save_configuration(path, config, name_model, name_config)

#     config_folder = path / name_model / "configurations"
#     assert config_folder.exists(), "Configuration folder was not created."
#     config_path = config_folder / f"{name_config}.yml"
#     assert config_path.exists(), "Configuration file was not saved."

#     with open(config_path, "r") as file:
#         saved_config = yaml.safe_load(file)
#     assert saved_config == config, "Saved configuration does not match the original."


# def test_save_tools(temp_dir, fake_optimizer, fake_scheduler):
#     path = temp_dir
#     name_model = "test_model"

#     save_tools(path, name_model, optimizer=fake_optimizer, scheduler=fake_scheduler)

#     tools_folder = path / name_model / "tools"
#     assert tools_folder.exists(), "Tools folder was not created."

#     optimizer_path = tools_folder / "optimizer.pth"
#     assert optimizer_path.exists(), "Optimizer state file was not saved."

#     scheduler_path = tools_folder / "scheduler.pth"
#     assert scheduler_path.exists(), "Scheduler state file was not saved."

#     # Verify optimizer state
#     loaded_optimizer_state = torch.load(optimizer_path)["optimizer_state_dict"]
#     assert loaded_optimizer_state == fake_optimizer.state_dict(), "Optimizer state does not match."

#     # Verify scheduler state
#     loaded_scheduler_state = torch.load(scheduler_path)["scheduler_state_dict"]
#     assert loaded_scheduler_state == fake_scheduler.state_dict(), "Scheduler state does not match."
