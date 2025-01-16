r"""Helper tools to perform loading operations"""

import torch
import yaml

from pathlib import Path

# isort: split
from poseidon.config import PATH_MODEL
from poseidon.data.const import DATASET_REGION, TOY_DATASET_REGION
from poseidon.diffusion.backbone import PoseidonBackbone


def load_backbone(
    name_model: str,
    path: Path = PATH_MODEL,
    best: bool = True,
    backup: bool = False,
) -> PoseidonBackbone:
    r"""Loads a `PoseidonBackbone` model.

    Arguments:
        name_model: Name of Backbone model to load.
        path: Path to folder in which save the model.
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
    )

    # Loading weights
    backbone_loaded.load_state_dict(model_ckpt["model_state_dict"])

    # By default, training mode
    return backbone_loaded.train()
