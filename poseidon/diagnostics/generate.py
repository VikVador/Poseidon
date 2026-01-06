r"""Tools to generate ensembles."""

import numpy as np
import os
import torch

from einops import rearrange
from torch import Tensor
from typing import Dict, Optional

# isort: split
from poseidon.config import PATH_DATA, PATH_EXP_MASKS, PATH_EXP_OBS, PATH_MODEL
from poseidon.data.datasets import PoseidonDataset
from poseidon.data.mappings import from_tensor_to_progressive_time
from poseidon.data.mask import generate_trajectory_mask
from poseidon.diffusion.coarsening import create_coarsen_variable
from poseidon.diffusion.denoiser import PoseidonDenoiser, PoseidonMMPSDenoiser
from poseidon.diffusion.observators import A_surface
from poseidon.diffusion.sampler import LMSSampler
from poseidon.diffusion.satellite import generate_satellite_surface_observation_model_parameters
from poseidon.diffusion.schedulers import PoseidonNoiseScheduler
from poseidon.diffusion.wrappers import PoseidonTrajectoryWrapper
from poseidon.training.load import load_backbone

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_reconstructions(config: Dict) -> None:
    r"""Generates reconstructions of reverse diffusion process.

    Arguments:
        config: Configuration for generation.
    """

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

    # Generating noise levels
    scheduler = PoseidonNoiseScheduler(config["sigma_min"], config["sigma_max"])
    noise_levels = scheduler(torch.linspace(0, 1, 32))

    # Loading mask of the Black Sea
    mask_bs = generate_trajectory_mask(trajectory_size=1)

    # Samples used for reconstruction estimation
    dates_samples = [
        ["2000-08-16", "2000-08-17"],
        ["2019-12-27", "2019-12-28"],
    ]

    for dates, split in zip(dates_samples, ["training", "validation"]):
        # Access to main folder
        path_folder = PATH_MODEL / config["model"] / "denoising" / split
        if not os.path.exists(path_folder):
            os.makedirs(path_folder)

        fname_x_truth = path_folder / "sample_truth.pt"
        fname_x_noisy = path_folder / "sample_noisy.pt"
        fname_x_recon = path_folder / "sample_reconstruction.pt"
        fname_noise = path_folder / "noise_levels.pt"

        # Getting the data
        x, time = next(
            iter(
                PoseidonDataset(
                    path=PATH_DATA,
                    date_start=dates[0],
                    date_end=dates[1],
                )
            )
        )

        # Saving the truth and noise levels
        torch.save(x, fname_x_truth)
        torch.save(noise_levels, fname_noise)

        # Adding batch dimension
        x, time = x.unsqueeze(0), time.unsqueeze(0)

        # Stores noisy states and reconstructions
        x_t_list, x_recon_list = [], []

        with torch.no_grad():
            for sigma_t in noise_levels:
                # Creating noisy state with initial signal in it
                x_t = sigma_t[None, None, None, None, None] * torch.randn_like(x) + x
                x_t_list.append(x_t.cpu())

                # Pushing to GPU
                x_t, sigma_t, time = (
                    x_t.to(DEVICE),
                    sigma_t[None, None].to(DEVICE),
                    time.to(DEVICE),
                )

                # State reconstruction estimation
                x_recon = rearrange(
                    model(x_t=rearrange(x_t, "B ... -> B (...)"), sigma_t=sigma_t, cond=time),
                    "B (C K X Y) -> B C K X Y",
                    C=C,
                    K=K,
                    X=X,
                    Y=Y,
                ).to("cpu")

                # Hiding the land
                x_recon[mask_bs == 0] = np.nan
                x_recon_list.append(x_recon)

            # Saving the results
            torch.save(torch.concat(x_t_list, dim=0), fname_x_noisy)
            torch.save(torch.concat(x_recon_list, dim=0), fname_x_recon)


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


def generate_from_posterior(date: str, config: Dict, y: Optional[Tensor] = None):
    r"""Generates a conditional ensemble.

    Arguments:
        date: Ensemble date (YYYY-MM-DD).
        config: Configuration for generation.
        y: Observations vector. If None, generates synthetic observation from ground truth.
    """

    # Access to model folder
    path_folder = (
        PATH_MODEL / config["model"] / "generation" / "posterior" / "experiments" / date
        if y is not None
        else PATH_MODEL / config["model"] / "generation" / "posterior" / date
    )

    if not os.path.exists(path_folder):
        os.makedirs(path_folder)

    # Name of the file
    fname = path_folder / "ensemble_posterior.pt"

    # Generating mean and covariance matrices of satellite observation model
    mu_y, cov_y = generate_satellite_surface_observation_model_parameters(
        observation_date=None if y is None else date,
        device=DEVICE,
    )

    # Creating observation operator
    A = A_surface(
        mu_y=mu_y,
        unscale=True,
        observation_date=None if y is None else date,
    )

    # Loading ground truth sample
    x, time = PoseidonDataset(
        path=PATH_DATA,
        date_start=date,
        date_end=date,
    )[0]

    # Pushing everything to GPU
    x, time = x.to(DEVICE), time.unsqueeze(0).to(DEVICE)

    # Extracting mean and standard deviation of observation model at observed locations
    _, sigma_x_obs = A.get_observation_statistics()

    # Rescaling the satellite observation covariance to model space
    cov_y = cov_y / (sigma_x_obs.to(DEVICE) ** 2)

    # If no observations provided, generate synthetic observations
    if y is None:
        y = A(x)
        y = y + torch.randn_like(y) * torch.sqrt(cov_y)
        y = y.to(DEVICE)

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
        A=A,
        cov_y=cov_y,
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

    # Generating an ensemble
    ensemble = sampler.forward(
        trajectory_size=config["trajectory_size"],
        ensemble_size=config["members"],
        steps=config["steps"],
        conditioning=time,
    ).cpu()

    # Loading Black Sea mask
    mask_bs = generate_trajectory_mask(trajectory_size=K)

    # Masking the nowcast
    ensemble[:, mask_bs[0] == 0] = np.nan

    # Saving the nowcast
    torch.save(ensemble, fname)


def generate_from_observations(date: str, config: Dict):
    r"""Generates a conditional ensemble using real observations.

    Arguments:
        date: Real observation date (YYYY-MM-DD).
        config: Configuration for generation.
    """

    # Defining paths to real observations coordinates
    path_coordinates = PATH_EXP_MASKS / "coordinates"

    # Loading real observations (nothing for SSH and chlorophyll needs to be coarsened)
    obs_sal = torch.load(PATH_EXP_OBS / f"{date}/sea_surface_salinity.pt", weights_only=False)
    obs_temp = torch.load(PATH_EXP_OBS / f"{date}/sea_surface_temperature.pt", weights_only=False)
    obs_chl = create_coarsen_variable(
        input_tensor=torch.from_numpy(torch.load(PATH_EXP_OBS / f"{date}/chlorophyll.pt", weights_only=False)).float(),
        lon_src=torch.load(path_coordinates / "longitude_316_455.pt", weights_only=False),
        lat_src=torch.load(path_coordinates / "latitude_316_455.pt", weights_only=False),
        lon_tgt=torch.load(path_coordinates / "longitude_128_256.pt", weights_only=False),
        lat_tgt=torch.load(path_coordinates / "latitude_128_256.pt", weights_only=False),
        target_resolution=(128, 256),
    )

    # Loading observation masks
    mask_chl = torch.load(PATH_EXP_MASKS / f"{date}/mask_chlorophyll.pt", weights_only=False)
    mask_sal = torch.load(PATH_EXP_MASKS / f"{date}/mask_salinity.pt", weights_only=False)
    mask_temp = torch.load(PATH_EXP_MASKS / f"{date}/mask_temperature.pt", weights_only=False)

    # Extract observation values at observed locations (where mask == 1)
    y_chl = obs_chl[mask_chl == 1].float()
    y_sal = torch.from_numpy(obs_sal[mask_sal == 1]).float()
    y_temp = torch.from_numpy(obs_temp[mask_temp == 1]).float()

    # Concatenate in the correct order (CHL, salinity, temperature)
    y_real = torch.concat([y_chl, y_sal, y_temp], dim=0).to(DEVICE)

    # Generating from posterior using true observations
    generate_from_posterior(date=date, config=config, y=y_real)
