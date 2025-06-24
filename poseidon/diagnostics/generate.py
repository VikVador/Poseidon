r"""Tools to generate (un)conditionnal nowcasts."""

import numpy as np
import os
import torch

from typing import Dict

# isort: split
from poseidon.config import PATH_MODEL
from poseidon.data.const import (
    DATASET_REGION,
    DATASET_VARIABLES,
    TOY_DATASET_REGION,
    TOY_DATASET_VARIABLES,
)
from poseidon.data.mask import generate_trajectory_mask
from poseidon.diffusion.denoiser import PoseidonDenoiser
from poseidon.diffusion.sampler import LMSSampler
from poseidon.diffusion.schedulers import PoseidonNoiseScheduler
from poseidon.diffusion.tools import PoseidonTrajectoryWrapper
from poseidon.training.load import load_backbone

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_unconditional_nowcast(index: int, config: Dict) -> None:
    """Generates an unconditional nowcast and saves it.

    Information:
        This function should be paired with a script that launches the generation on multiple GPUs.

    Arguments:
        index: ID of nowcast to generate.
        config: Configuration for generation.
    """

    # Path to the model
    path_folder = PATH_MODEL / config["model"] / "nowcasts" / "unconditional"

    # Creating path to saving folder
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)

    # Name of the nowcast to save
    fname = path_folder / f"nowcast_unconditional_{index}.pt"

    # Loading mask of the Black Sea
    mask_bs = generate_trajectory_mask(
        variables=TOY_DATASET_VARIABLES if config["toy_problem"] else DATASET_VARIABLES,
        region=TOY_DATASET_REGION if config["toy_problem"] else DATASET_REGION,
        trajectory_size=1,
    )

    # Loading the neural network denoiser
    model = (
        PoseidonDenoiser(
            backbone=load_backbone(name_model=config["model"], best=config["best"]),
        )
        .eval()
        .to(DEVICE)
    )

    # Dimensions of the problem
    C, K, X, Y = (
        model.backbone.C,
        model.backbone.K,
        model.backbone.X,
        model.backbone.Y,
    )

    # Wrapping the model for trajectory generation
    model = PoseidonTrajectoryWrapper(
        model,
        dimensions=(C, X, Y),
        blanket_size=K,
    )

    # Creating the sampler
    sampler = LMSSampler(
        denoiser=model,
        schedule=PoseidonNoiseScheduler(),
        dimensions=(C, X, Y),
        order=3,
    )

    # Generating a nowcast
    nowcast = sampler.forward(
        trajectory_size=1,
        forecast_size=1,
        steps=config["steps"],
    ).cpu()

    # Masking the nowcast
    nowcast[:, mask_bs[0] == 0] = np.nan

    # Saving the nowcast
    torch.save(nowcast, fname)
