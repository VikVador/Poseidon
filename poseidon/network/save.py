r"""Network - Tools to save and load backbones."""

import os
import random
import string
import torch
import yaml

from pathlib import Path
from typing import Dict

# isort: split
from poseidon.config import POSEIDON_MODEL
from poseidon.data.const import DATASET_REGION, TOY_DATASET_REGION
from poseidon.diffusion.backbone import PoseidonBackbone


def generate_model_name(length: int) -> str:
    r"""Helper tool to generate a random alphanumeric string.

    Arguments:
        length: Length of the name
    """
    characters = string.ascii_letters + string.digits
    return "".join(random.choice(characters) for _ in range(length))


def save_configuration(path: Path, configs: Dict, verbose: bool = True) -> None:
    r"""Saves a configuration as a .yml file.

    Arguments:
        path: Path to save the configuration.
        configs: Dictionary containing a given configuration.
        verbose: Whether or not display information.
    """
    config_file = path / "training_config.yml"
    with open(config_file, "w") as file:
        yaml.dump(configs, file)
    if verbose:
        print(f"Configuration Saved | File: {config_file}")


def save_backbone(
    path: Path, model: PoseidonBackbone, optimizer: torch.optim, epoch: int, verbose: bool = True
) -> None:
    r"""Saves a backbone model.

    Arguments:
        path: Path to save the model.
        model: Backbone to save in its current state.
        optimizer: Optimizer to save.
        epoch: Epoch at which the model is saved.
        verbose: Whether or not display information about the saved model.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    save_file = os.path.join(path, f"checkpoint_{epoch}.pth")
    torch.save(checkpoint, save_file)
    if verbose:
        print(f"Model Saved | Epoch: {epoch} - File: {save_file}")


def load_backbone(name: str, checkpoint: int) -> PoseidonBackbone:
    r"""Loads a backbone model from a given checkpoint.

    Arguments:
        name: Name of the backbone to load.
        checkpoint: Index of backbone state to load.
    """

    folder = os.path.join(POSEIDON_MODEL, name)
    file_config = os.path.join(folder, "training_config.yml")
    file_checkpoint = os.path.join(folder, f"checkpoint_{checkpoint}.pth")
    with open(file_config, "r") as file:
        configs = yaml.load(file, Loader=yaml.Loader)

    # Configuring the backbone
    backbone = PoseidonBackbone(
        **configs["config_backbone"],
        dimensions=(
            configs["config_problem"]["Channels"],
            configs["config_problem"]["Trajectory Size"],
            configs["config_problem"]["Latitudes"],
            configs["config_problem"]["Longitudes"],
        ),
        config_nn=configs["config_nn"],
        config_region=TOY_DATASET_REGION
        if configs["config_problem"]["Toy_problem"]
        else DATASET_REGION,
    )

    # Restoring the model
    checkpoint = torch.load(file_checkpoint, map_location="cpu")
    backbone.load_state_dict(checkpoint["model_state_dict"])
    backbone.train()
    return backbone
