r"""Satellite observation models parameters."""

import torch

from torch import Tensor
from typing import Tuple

# isort: split
# fmt: off
from poseidon.data.mask import generate_trajectory_mask
from poseidon.diffusion import (
    SAT_CHL_MU,
    SAT_CHL_STD,
    SAT_SAL_MU,
    SAT_SAL_STD,
    SAT_SSH_MU,
    SAT_SSH_STD,
    SAT_TEMP_MU,
    SAT_TEMP_STD,
)


def sample_parameters(mean: float, std: float) -> Tensor:
    """Sample parameters from a uniform distribution."""
    if std == 0.0:
        return torch.tensor(mean)
    else:
        a, b = mean - std, mean + std
        return torch.tensor(torch.rand(1).item() * (b - a) + a)


def generate_satellite_observation_parameters() -> (
    Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
):
    """Generates realistic parameters for satellite observation model

    Source:
        Copernicus Marine Environment Monitoring Service (CMEMS)

    Mathematical model:
        y ~ N(A(x) + mu_y, sigma_y ** 2)

    Returns:
        Mean and standard deviation of observation model.
    """

    # Sampling parameters
    mu_chl, sigma_chl = (
        sample_parameters(mean=SAT_CHL_MU[0], std=SAT_CHL_MU[1]),
        sample_parameters(mean=SAT_CHL_STD[0], std=SAT_CHL_STD[1]),
    )

    mu_sal, sigma_sal = (
        sample_parameters(mean=SAT_SAL_MU[0], std=SAT_SAL_MU[1]),
        sample_parameters(mean=SAT_SAL_STD[0], std=SAT_SAL_STD[1]),
    )

    mu_temp, sigma_temp = (
        sample_parameters(mean=SAT_TEMP_MU[0], std=SAT_TEMP_MU[1]),
        sample_parameters(mean=SAT_TEMP_STD[0], std=SAT_TEMP_STD[1]),
    )

    mu_ssh, sigma_ssh = (
        sample_parameters(mean=SAT_SSH_MU[0], std=SAT_SSH_MU[1]),
        sample_parameters(mean=SAT_SSH_STD[0], std=SAT_SSH_STD[1]),
    )

    return mu_chl, sigma_chl, mu_sal, sigma_sal, mu_temp, sigma_temp, mu_ssh, sigma_ssh


def generate_satellite_surface_observation_model_parameters() -> Tuple[Tensor, Tensor]:
    r"""Generates the mean and covariance matrices for a satellite observing the surface.

    Information:
        We only observe chlorophyll, salinity, temperature, sea surface height are observed.
    """

    # Indices of variables at the surface (chl, sal, temp, ssh)
    indices = [32, 64, 96, 128]

    # Generating observation parameters
    mu_chl, sigma_chl, mu_sal, sigma_sal, mu_temp, sigma_temp, mu_ssh, sigma_ssh = (
        generate_satellite_observation_parameters()
    )

    # Loading mask of the Black Sea
    mask = generate_trajectory_mask(trajectory_size=1)[0]

    # Creating mean and covariance matrices
    mu_y, cov_y, mask = (
        mask[indices].clone() * torch.tensor([mu_chl, mu_sal, mu_temp, mu_ssh]).view(-1, 1, 1, 1),
        mask[indices].clone() * torch.tensor([sigma_chl**2, sigma_sal**2, sigma_temp**2, sigma_ssh**2]).view(-1, 1, 1, 1),
        mask[indices].clone(),
    )

    # Extracting values for observed locations
    mu_y, cov_y = mu_y[mask == 1], cov_y[mask == 1]
    mu_y.requires_grad  = True
    cov_y.requires_grad = True

    return mu_y, cov_y
