r"""Satellite observation models parameters."""

import torch

from torch import Tensor
from typing import Optional, Tuple

# isort: split
from poseidon.config import PATH_EXP_MASKS
from poseidon.data.const import OBSERVATIONS_RESOLUTION
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


# fmt: off
def sample_parameters(mean: float, std: float) -> Tensor:
    r"""Sample parameters from a uniform distribution."""
    if std == 0.0:
        return torch.tensor(mean)
    else:
        a, b = mean - std, mean + std
        return torch.tensor(torch.rand(1).item() * (b - a) + a)


def generate_satellite_observation_parameters() -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    r"""Generates realistic parameters for satellite observation model

    Mathematical model:
        y ~ N(A(x) + mu_y, sigma_y ** 2)

    Returns:
        Mean (mu_y) and standard deviation (sigma_y) of observation model for chlorophyll, salinity, temperature, and sea surface height.
    """

    # Sampling parameters
    mu_chl, sigma_chl = (
        sample_parameters(mean=SAT_CHL_MU[0],  std=SAT_CHL_MU[1]),
        sample_parameters(mean=SAT_CHL_STD[0], std=SAT_CHL_STD[1]),
    )

    mu_sal, sigma_sal = (
        sample_parameters(mean=SAT_SAL_MU[0],  std=SAT_SAL_MU[1]),
        sample_parameters(mean=SAT_SAL_STD[0], std=SAT_SAL_STD[1]),
    )

    mu_temp, sigma_temp = (
        sample_parameters(mean=SAT_TEMP_MU[0],  std=SAT_TEMP_MU[1]),
        sample_parameters(mean=SAT_TEMP_STD[0], std=SAT_TEMP_STD[1]),
    )

    mu_ssh, sigma_ssh = (
        sample_parameters(mean=SAT_SSH_MU[0],  std=SAT_SSH_MU[1]),
        sample_parameters(mean=SAT_SSH_STD[0], std=SAT_SSH_STD[1]),
    )

    return mu_chl, sigma_chl, mu_sal, sigma_sal, mu_temp, sigma_temp, mu_ssh, sigma_ssh


def generate_satellite_surface_observation_model_parameters(observation_date: Optional[str] = None, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> Tuple[Tensor, Tensor]:
    r"""Generates the mean and covariance vectors for a satellite observing the surface.

    Note:
        If no observation_date is provided, we assume synthetic observations and include sea surface height.

    Arguments:
        observation_date: Date of real observations (YYYY-MM-DD).
        device: Device for tensor operations.
    """

    # Variable names in order
    variable_names = ["CHL", "vosaline", "votemper"] + (["ssh"] if observation_date is None else [])

    # Generating observation parameters
    mu_chl, sigma_chl, mu_sal, sigma_sal, mu_temp, sigma_temp, mu_ssh, sigma_ssh = (
        generate_satellite_observation_parameters()
    )

    # Store parameters in order matching variable_names
    mu_params    = [mu_chl, mu_sal, mu_temp]          + ([mu_ssh]    if observation_date is None else [])
    sigma_params = [sigma_chl, sigma_sal, sigma_temp] + ([sigma_ssh] if observation_date is None else [])

    # Define paths
    path_coarsened = PATH_EXP_MASKS / "coarsened"

    # Loading masks for each observed variable
    if observation_date is None:
        # Synthetic observations: use coarsened masks
        coarsened_masks = {
            var: torch.load(
                path_coarsened / f"mask_{OBSERVATIONS_RESOLUTION[var.lower()][0]}_{OBSERVATIONS_RESOLUTION[var.lower()][1]}.pt",
                weights_only=False,
                map_location=device
            )[0].values
            for var in variable_names
        }
    else:
        # Real observations: use date-specific masks
        path_obs_masks = PATH_EXP_MASKS / observation_date
        mask_files = {"CHL": "mask_chlorophyll.pt", "vosaline": "mask_salinity.pt", "votemper": "mask_temperature.pt"}
        coarsened_masks = {
            var: torch.load(path_obs_masks / mask_files[var], weights_only=False, map_location=device)
            for var in variable_names
        }

    # Compute observation vector sizes from masks
    obs_sizes = [
        int(torch.sum(torch.tensor(coarsened_masks[var], device=device) == 1).item())
        for var in variable_names
    ]

    # Create observation model parameter vectors
    mu_y_list = [
        torch.ones(n, device=device) * mu
        for n, mu in zip(obs_sizes, mu_params)
    ]

    cov_y_list = [
        torch.ones(n, device=device) * (sigma ** 2)
        for n, sigma in zip(obs_sizes, sigma_params)
    ]

    # Concatenate to match observation vector structure
    mu_y  = torch.concat(mu_y_list, dim=0)
    cov_y = torch.concat(cov_y_list, dim=0)

    mu_y.requires_grad = True
    cov_y.requires_grad = True

    return mu_y, cov_y
