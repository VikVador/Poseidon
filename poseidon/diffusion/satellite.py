r"""Satellite Observations Parameters."""

import numpy as np
import torch
import xarray as xr

from typing import Tuple

# isort: split
from poseidon.config import PATH_STAT
from poseidon.data.const import DATASET_REGION
from poseidon.data.mask import generate_trajectory_mask
from poseidon.diffusion import (
    SAT_CHL_BIAS,
    SAT_CHL_STD,
    SAT_SAL_BIAS,
    SAT_SAL_STD,
    SAT_SSH_BIAS,
    SAT_SSH_STD,
    SAT_TEMP_BIAS,
    SAT_TEMP_STD,
)


def sample_parameters(mean: float, std: float) -> np.ndarray:
    """Sample parameters from a uniform distribution."""
    if std == 0.0:
        return torch.tensor(mean)
    else:
        a, b = mean - std, mean + std
        return torch.tensor(torch.rand(1).item() * (b - a) + a)


def generate_satellite_error_parameters() -> (
    Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]
):
    """Generates (realistics) satellite error parameters for standardized data."""

    # Opening statistics dataset
    stats = xr.open_zarr(PATH_STAT).isel(level=DATASET_REGION["level"]).load()

    # Stats for standardization
    stats_std = stats.isel(level=0).sel(statistic="std")
    stats_std_temps = stats_std["votemper"].item()
    stats_std_chls = stats_std["CHL"].item()
    stats_std_sshs = stats_std["ssh"].item()
    stats_std_sals = stats_std["vosaline"].item()

    # Applying standardization
    bias_temp, std_temp = (
        [bt / stats_std_temps for bt in SAT_TEMP_BIAS],
        [st / stats_std_temps for st in SAT_TEMP_STD],
    )

    bias_chl, std_chl = (
        [bc / stats_std_chls for bc in SAT_CHL_BIAS],
        [sc / stats_std_chls for sc in SAT_CHL_STD],
    )

    bias_ssh, std_ssh = (
        [bs / stats_std_sshs for bs in SAT_SSH_BIAS],
        [ss / stats_std_sshs for ss in SAT_SSH_STD],
    )

    bias_sal, std_sal = (
        [bs / stats_std_sals for bs in SAT_SAL_BIAS],
        [ss / stats_std_sals for ss in SAT_SAL_STD],
    )

    # Sampling parameters
    mu_chl, sigma_chl = (
        sample_parameters(mean=bias_chl[0], std=bias_chl[1]),
        sample_parameters(mean=std_chl[0], std=std_chl[1]),
    )

    mu_sal, sigma_sal = (
        sample_parameters(mean=bias_sal[0], std=bias_sal[1]),
        sample_parameters(mean=std_sal[0], std=std_sal[1]),
    )

    mu_temp, sigma_temp = (
        sample_parameters(mean=bias_temp[0], std=bias_temp[1]),
        sample_parameters(mean=std_temp[0], std=std_temp[1]),
    )

    mu_ssh, sigma_ssh = (
        sample_parameters(mean=bias_ssh[0], std=bias_ssh[1]),
        sample_parameters(mean=std_ssh[0], std=std_ssh[1]),
    )

    return mu_chl, sigma_chl, mu_sal, sigma_sal, mu_temp, sigma_temp, mu_ssh, sigma_ssh


def generate_satellite_gaussian_parameters():
    r"""Generates the mean and covariance matrices for satellite surface observations."""

    # Generating observation parameters
    mu_chl, sigma_chl, mu_sal, sigma_sal, mu_temp, sigma_temp, mu_ssh, sigma_ssh = generate_satellite_error_parameters()

    # Indices of surface variables (chl, sal, temp, ssh)
    indices = [32, 64, 96, 128]

    # Loading mask of the Black Sea
    mask = generate_trajectory_mask(trajectory_size=1)[0]

    # Computing mean and covariance of the observations
    mu_y, cov_y, mask = (
        mask[indices].clone() * torch.tensor([mu_chl, mu_sal, mu_temp, mu_ssh]).view(-1, 1, 1, 1),
        mask[indices].clone() * torch.tensor([sigma_chl, sigma_sal, sigma_temp, sigma_ssh]).view(-1, 1, 1, 1),
        mask[indices].clone(),
    )

    # Extracting at observed locations
    mu_y, cov_y = (
        mu_y[mask == 1],
        cov_y[mask == 1],
    )

    # Activating autograd
    mu_y.requires_grad = True
    cov_y.requires_grad = True

    return mu_y, cov_y
