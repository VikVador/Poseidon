r"""Observation operators."""

import numpy as np
import torch
import torch.nn as nn
import xarray as xr

from torch import Tensor
from typing import Tuple

# isort: split
# fmt: off
from poseidon.config import PATH_STAT
from poseidon.data.const import DATASET_REGION, DATASET_VARIABLES, DATASET_VARIABLES_OCEAN
from poseidon.data.mask import generate_trajectory_mask

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class A_surface(nn.Module):
    r"""An observation operator for the surface.

    Notes:
        If the physical bias of the observation model is in physical units, unscale must be set to True.

    Arguments:
        mu_y: Mean bias of the observation model.
        unscale: Whether or not to unscale x ~ E(x|xt) before applying the observation operator.
    """

    def __init__(self, mu_y: Tensor = None, unscale: bool = True):
        super().__init__()

        # Indices of variables at the surface (chl, sal, temp, ssh)
        self.indices = [32, 64, 96, 128]

        # Loading mask of the Black Sea
        self.mask = generate_trajectory_mask(trajectory_size=1)[0].to(DEVICE)

        # Loading statistics
        ds_mean = (xr.open_zarr(PATH_STAT).isel(level=DATASET_REGION["level"]).sel(statistic="mean")[DATASET_VARIABLES].load())
        ds_std  = (xr.open_zarr(PATH_STAT).isel(level=DATASET_REGION["level"]).sel(statistic="std")[DATASET_VARIABLES].load())

       # Saving mean bias of the observation model (Physical units)
        self.mu_y = mu_y

        # Determine wether or not to unscale x ~ E(x|xt) before applying the observation operator
        self.unscaling = unscale

        # Storing mean and standard deviation of training dataset
        self.mu_x = (
            torch.concat(
                [torch.from_numpy(ds_mean[var].values) for var in DATASET_VARIABLES_OCEAN]
                + [torch.tensor([ds_mean["ssh"].values[0]])],
                dim=0,
            )
            .view(-1, 1, 1, 1)
            .to(DEVICE)
        )

        self.sigma_x = (
            torch.concat(
                [torch.from_numpy(ds_std[var].values) for var in DATASET_VARIABLES_OCEAN]
                + [torch.tensor([ds_std["ssh"].values[0]])],
                dim=0,
            )
            .view(-1, 1, 1, 1)
            .to(DEVICE)
        )

        # Storing mean and standard deviation of training dataset at observed locations
        self.mask_obs    = self.mask[self.indices, 0, :, :]
        self.mu_x_obs    = (self.mask_obs *    self.mu_x[self.indices, 0, :, :])[self.mask_obs == 1]
        self.sigma_x_obs = (self.mask_obs * self.sigma_x[self.indices, 0, :, :])[self.mask_obs == 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Applies the observation operator."""

        # Unscaling x ~ E(x|xt)
        x = self.unscale(x)

        # Extracting surface values
        x = x[self.indices, 0, :, :]

        # Extracting sea values
        x = x[self.mask_obs == 1]

        # Adding physical bias
        x = x + self.mu_y if self.mu_y is not None else x

        # Scaling to standardized space
        return self.scale(x)

    def visualize(self, x: torch.Tensor) -> torch.Tensor:
        r"""Visualizes observation operator effect."""

        # Extracting surface values
        x = x[self.indices, 0, :, :]

        # Hiding land values
        x[self.mask_obs == 0] = np.nan

        return x

    def unscale(self, x: Tensor) -> Tensor:
        r"""Unstandardizes the input tensor."""
        return x * self.sigma_x + self.mu_x if self.unscaling else x

    def scale(self, x: Tensor) -> Tensor:
        r"""Standardizes input tensor."""
        return (x - self.mu_x_obs) / self.sigma_x_obs if self.unscaling else x

    def get_observation_statistics(self) -> Tuple[Tensor, Tensor]:
        r"""Returns training dataset mean and standard deviation at observed locations."""
        return self.mu_x_obs, self.sigma_x_obs
