r"""Tools to generate nowcasts."""

import numpy as np
import os
import torch

from typing import Dict

# isort: split
from poseidon.config import PATH_DATA, PATH_MODEL
from poseidon.data.datasets import PoseidonDataset
from poseidon.data.mappings import from_tensor_to_progressive_time
from poseidon.data.mask import generate_trajectory_mask
from poseidon.diffusion.denoiser import PoseidonDenoiser, PoseidonMMPSDenoiser
from poseidon.diffusion.observators import A_surface
from poseidon.diffusion.sampler import LMSSampler
from poseidon.diffusion.schedulers import PoseidonNoiseScheduler
from poseidon.diffusion.wrappers import PoseidonTrajectoryWrapper
from poseidon.training.load import load_backbone

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_from_prior(date: str, config: Dict) -> None:
    r"""Generates an unconditional ensemble.

    Arguments:
        date: Ensemble date (MM-DD).
        config: Configuration for generation.
    """

    # Access to model folder
    path_folder = PATH_MODEL / config["model"] / "generation" / "prior" / date
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)

    # Name of the file
    fname = path_folder / "ensemble_prior.pt"

    # Loading corresponding model
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

    # Generating a random conditioning (i.e. year progression for a nowcasting)
    conditioning = (
        from_tensor_to_progressive_time(torch.tensor([[2000, int(date[:2]), int(date[3:]), 12]]))
        .unsqueeze(0)
        .to(DEVICE)
    )

    # Generating an ensemble
    ensemble = sampler.forward(
        trajectory_size=K,
        ensemble_size=config["members"],
        steps=config["steps"],
        conditioning=conditioning,
    ).cpu()

    # Loading Black Sea mask
    mask_bs = generate_trajectory_mask(trajectory_size=K)

    # Masking the nowcast
    ensemble[:, mask_bs[0] == 0] = np.nan

    # Saving the nowcast
    torch.save(ensemble, fname)


def generate_from_posterior(date: str, config: Dict) -> None:
    r"""Generates a conditional ensemble.

    Arguments:
        date: Ensemble date (YYYY-MM-DD).
        config: Configuration for generation.
    """

    # Access to model folder
    path_folder = PATH_MODEL / config["model"] / "generation" / "posterior" / date
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)

    # Name of the file
    fname = path_folder / "ensemble_posterior.pt"

    # Loading ground truth sample
    x, time = PoseidonDataset(
        path=PATH_DATA,
        date_start=date,
        date_end=date,
    )[0]

    # Observation model used to generate the nowcast
    observator = A_surface()

    # Generating observation
    y = observator(x)

    # Pushing to GPU
    y, conditioning = y.cuda(), time.unsqueeze(0).cuda()

    # Loading corresponding model
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
        tweedie_covariance=["tweedie_covariance"],
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

    # Generating an ensemble
    ensemble = sampler.forward(
        trajectory_size=config["trajectory_size"],
        ensemble_size=config["members"],
        steps=config["steps"],
        conditioning=conditioning,
    ).cpu()

    # Loading Black Sea mask
    mask_bs = generate_trajectory_mask(trajectory_size=K)

    # Masking the nowcast
    ensemble[:, mask_bs[0] == 0] = np.nan

    # Saving the nowcast
    torch.save(ensemble, fname)
