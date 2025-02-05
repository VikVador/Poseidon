r"""Helper tools to load checkpoints and models."""

import torch
import yaml

from pathlib import Path
from torch.optim import Optimizer, lr_scheduler

# isort: split
from poseidon.config import PATH_MESH, PATH_MODEL
from poseidon.data.const import DATASET_REGION, TOY_DATASET_REGION
from poseidon.diffusion.backbone import PoseidonBackbone


def load_backbone(
    name_model: str,
    path: Path = PATH_MODEL,
    path_mesh: Path = PATH_MESH,
    best: bool = True,
    backup: bool = False,
) -> PoseidonBackbone:
    r"""Loads a `PoseidonBackbone` model.

    Arguments:
        name_model: Name of Backbone model to load.
        path: Path to folder in which save the model.
        path_mesh: Path to the mesh file used by the backbone.
        best: Weather to load the best model or the last one.
        backup: Weather to load the backup model.
    """

    # Loading model state
    model_path = (
        path
        / name_model
        / ("__backup__/models" if backup else "models")
        / ("best.pth" if best else "last.pth")
    )

    model_ckpt = torch.load(model_path, weights_only="True", map_location="cpu")

    # Loading configuration files
    path_cfg = path / name_model / "configurations"
    path_cfg_dimensions = path_cfg / "dimensions.yml"
    path_cfg_problem = path_cfg / "problem.yml"
    path_cfg_unet = path_cfg / "unet.yml"
    path_cfg_siren = path_cfg / "siren.yml"

    with open(path_cfg_dimensions, "r") as file:
        dimensions = yaml.safe_load(file)

    with open(path_cfg_problem, "r") as file:
        problem = yaml.safe_load(file)

    with open(path_cfg_unet, "r") as file:
        unet = yaml.safe_load(file)

    with open(path_cfg_siren, "r") as file:
        siren = yaml.safe_load(file)

    # Loading backbone
    backbone_loaded = PoseidonBackbone(
        dimensions=dimensions["dimensions"],
        config_unet=unet,
        config_siren=siren,
        config_region=TOY_DATASET_REGION if problem["toy_problem"] else DATASET_REGION,
        path_mesh=path_mesh,
    )

    # Loading weights
    backbone_loaded.load_state_dict(model_ckpt["model_state_dict"])

    # By default, training mode
    return backbone_loaded.train()


def load_optimizer(
    name_model: str,
    optimizer: Optimizer,
    path: Path = PATH_MODEL,
) -> None:
    r"""Loads an optimizer state from a saved file.

    Arguments:
        name_model: Name of the model trained with this tool.
        optimizer: Optimizer instance to load the state into.
        path: Path to folder containing saved optimizer.
    """

    path_tool = path / f"{name_model}" / "tools" / "optimizer.pth"
    assert path_tool.exists(), f"ERROR - Optimizer file not found at {path_tool}"

    # Loading the checkpoint
    checkpoint = torch.load(path_tool, weights_only=True)

    # Loading the  state in current optimizer (in-place)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


def load_scheduler(
    name_model: str,
    scheduler: lr_scheduler,
    path: Path = PATH_MODEL,
) -> None:
    r"""Loads a scheduler state from a saved file.

    Arguments:
        name_model: Name of the model trained with this tool.
        scheduler: Scheduler instance to load the state into.
        path: Path to folder containing saved scheduler.
    """
    path_tool = path / f"{name_model}" / "tools" / "scheduler.pth"
    assert path_tool.exists(), f"ERROR - Scheduler file not found at {path_tool}"

    # Loading the checkpoint
    checkpoint = torch.load(path_tool, weights_only=True)

    # Loading the state in current scheduler (in-place)
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
