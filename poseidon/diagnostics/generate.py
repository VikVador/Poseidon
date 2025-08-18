r"""Tools to generate nowcasts."""

import numpy as np
import torch

from typing import Dict

# isort: split
from poseidon.config import PATH_DATA, PATH_MODEL
from poseidon.data.const import (
    DATASET_REGION,
    DATASET_VARIABLES,
    TOY_DATASET_REGION,
    TOY_DATASET_VARIABLES,
)
from poseidon.data.datasets import PoseidonDataset
from poseidon.data.mask import generate_trajectory_mask
from poseidon.diagnostics.const import DATES_POSTERIOR_MONTHLY
from poseidon.diagnostics.tools import create_day_index_mapping
from poseidon.diffusion.denoiser import PoseidonDenoiser, PoseidonMMPSDenoiser
from poseidon.diffusion.observators import A_surface
from poseidon.diffusion.sampler import LMSSampler
from poseidon.diffusion.schedulers import PoseidonNoiseScheduler
from poseidon.diffusion.tools import PoseidonTrajectoryWrapper
from poseidon.training.load import load_backbone

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_unconditional(index: int, config: Dict, date: str = None) -> None:
    r"""Generates an unconditional nowcast and saves it.

    Information:
        This function should be paired with a script that launches the generation on multiple GPUs.

    Arguments:
        index: ID of nowcast to generate.
        config: Configuration for generation.
        date: Date for which to generate the nowcast (YYYY-MM-DD).
    """

    # Path to the model
    path_folder = PATH_MODEL / config["model"] / "nowcasts" / "unconditional"

    # Additionnal folder
    path_folder = path_folder / "random" if date is None else path_folder / date

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
        schedule=PoseidonNoiseScheduler(
            sigma_min=config["sigma_min"],
            sigma_max=config["sigma_max"],
        ),
        dimensions=(C, X, Y),
        order=3,
    )

    # Used to create the mapping between dates and indexes
    map_d_i = create_day_index_mapping()

    # Generating a random conditioning (i.e. year progression for a nowcasting)
    conditioning = (
        torch.randint(1, 365, (1, 1)).to(DEVICE) / 365
        if date is None
        else torch.ones((1, 1)) * map_d_i[date[5:]] / 365
    )

    # Generating a nowcast
    nowcast = sampler.forward(
        trajectory_size=1,
        forecast_size=config["nb_nowcasts"],
        steps=config["steps"],
        conditioning=conditioning,
    ).cpu()

    # Masking the nowcast
    nowcast[:, mask_bs[0] == 0] = np.nan

    # Saving the nowcast
    torch.save(nowcast, fname)


def generate_conditional(index: int, config: Dict) -> None:
    r"""Generates conditional nowcasts and save them.

    Information:
        This function should be paired with a script that launches the generation on multiple GPUs.

    Arguments:
        index: Index of month from which sample a ground truth.
        config: Configuration for generation.
    """

    # Security
    assert 0 <= index <= 11, "ERROR - Index must be between 0 and 11 (inclusive)."

    # Initialization
    toy_problem = config["toy_problem"]
    region = TOY_DATASET_REGION if toy_problem else DATASET_REGION
    variables = TOY_DATASET_VARIABLES if toy_problem else DATASET_VARIABLES
    path_folder = PATH_MODEL / config["model"] / "nowcasts" / "conditional"

    # Name of the nowcast to save
    fname = path_folder / f"nowcast_conditional_{index}.pt"

    # Loading mask of the Black Sea
    mask_bs = generate_trajectory_mask(
        variables=variables,
        region=region,
        trajectory_size=1,
    )

    # Loading sample
    x, time = PoseidonDataset(
        path=PATH_DATA,
        date_start=DATES_POSTERIOR_MONTHLY[index],
        date_end=DATES_POSTERIOR_MONTHLY[index],
        variables=variables,
        region=region,
    )[0]

    # Observation model used to generate the nowcast
    observator = A_surface(
        variables=variables,
        region=region,
    )

    # Generating observation
    y = observator(x)

    # Pushing to GPU
    y, time = y.cuda(), time.unsqueeze(0).cuda()

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

    # Adding MMPS wrapper for conditional sampling
    model = PoseidonMMPSDenoiser(
        denoiser=model,
        y=y,
        A=observator,
        cov_y=config["covariance_y"],
        tweedie_covariance=True,
        iterations=config["iterations"],
    )

    # Creating the sampler
    sampler = LMSSampler(
        denoiser=model,
        schedule=PoseidonNoiseScheduler(
            sigma_min=config["sigma_min"],
            sigma_max=config["sigma_max"],
        ),
        dimensions=(C, X, Y),
        order=3,
    )

    # Generating a nowcast
    nowcast = sampler.forward(
        trajectory_size=1,
        forecast_size=config["nb_nowcasts"],
        steps=config["steps"],
        conditioning=time,
    )

    # Masking the nowcast
    nowcast[:, mask_bs[0] == 0] = np.nan

    # Saving the nowcast
    torch.save(nowcast, fname)
