r"""Observation operators."""

import numpy as np
import torch
import torch.nn as nn
import xarray as xr

from torch import Tensor
from typing import Optional, Tuple

# isort: split
from poseidon.config import PATH_EXP_MASKS, PATH_STAT
from poseidon.data.const import DATASET_REGION, DATASET_VARIABLES, DATASET_VARIABLES_OCEAN, OBSERVATIONS_RESOLUTION
from poseidon.diffusion.coarsening import create_coarsen_variable

# fmt: off
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class A_surface(nn.Module):
    r"""A realistic satellite observation operator.

    Note:
        1. For synthetic observations, it observes CHL, salinity, temperature, AND SSH.
        2. For real observations, it observes CHL, salinity, and temperature ONLY (no SSH).

    Arguments:
        mu_y: Mean bias of the observation model in physical units.
        unscale: Whether to unscale x ~ E(x|xt) before applying the observation operator.
        observation_date: Date of real observations (YYYY-MM-DD). If None, uses synthetic observations.
    """

    def __init__(
        self,
        unscale: bool = True,
        mu_y: Optional[Tensor] = None,
        observation_date: Optional[str] = None,
    ):
        super().__init__()

        self.observation_date = observation_date

        # Determine which variables to observe based on observation type
        if observation_date is None:
            # Synthetic observations
            self.indices = [32, 64, 96, 128]
            self.variable_names = ["CHL", "vosaline", "votemper", "ssh"]
        else:
            # Real observations
            self.indices = [32, 64, 96]
            self.variable_names = ["CHL", "vosaline", "votemper"]

        # Define paths
        path_coarsened = PATH_EXP_MASKS / "coarsened"
        path_coordinates = PATH_EXP_MASKS / "coordinates"

        # Loading masks for each observed variable
        if observation_date is None:
            # Synthetic observations: use coarsened masks
            self.coarsened_masks = {
                var: torch.load(
                    path_coarsened / f"mask_{OBSERVATIONS_RESOLUTION[var.lower()][0]}_{OBSERVATIONS_RESOLUTION[var.lower()][1]}.pt",
                    weights_only=False,
                    map_location=DEVICE
                )[0].values
                for var in self.variable_names
            }
        else:
            # Real observations: use date-specific masks (SSH not included)
            path_obs_masks = PATH_EXP_MASKS / observation_date
            mask_files = {"CHL": "mask_chlorophyll.pt", "vosaline": "mask_salinity.pt", "votemper": "mask_temperature.pt"}
            self.coarsened_masks = {
                var: torch.load(path_obs_masks / mask_files[var], weights_only=False, map_location=DEVICE)
                for var in self.variable_names
            }

        # Loading longitude and latitude coordinates of original mesh
        self.coordinates = {
            coord: torch.load(path_coordinates / f"{coord}_128_256.pt", weights_only=False, map_location=DEVICE)
            for coord in ["longitude", "latitude"]
        }

        # Loading longitude and latitude coordinates for each observed variable
        self.coarsened_coordinates = {
            var: {
                coord: torch.load(
                    path_coordinates / f"{coord}_{OBSERVATIONS_RESOLUTION[var.lower()][0]}_{OBSERVATIONS_RESOLUTION[var.lower()][1]}.pt",
                    weights_only=False,
                    map_location=DEVICE
                )
                for coord in ["longitude", "latitude"]
            }
            for var in self.variable_names
        }

        # Loading statistics
        ds_mean = xr.open_zarr(PATH_STAT).isel(level=DATASET_REGION["level"]).sel(statistic="mean")[DATASET_VARIABLES].load()
        ds_std  = xr.open_zarr(PATH_STAT).isel(level=DATASET_REGION["level"]).sel(statistic="std")[DATASET_VARIABLES].load()

        # Saving mean bias of the observation model
        self.mu_y = mu_y
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

        # Dynamically compute observation vector sizes from masks
        obs_sizes = [
            int(torch.sum(torch.tensor(self.coarsened_masks[var], device=DEVICE) == 1).item())
            for var in self.variable_names
        ]

        # Storing mean and standard deviation with shape of extracted observations
        self.mu_x_obs = torch.concat([
            torch.ones([n], device=DEVICE) * self.mu_x[idx, 0, 0, 0]
            for n, idx in zip(obs_sizes, self.indices)
        ])

        self.sigma_x_obs = torch.concat([
            torch.ones([n], device=DEVICE) * self.sigma_x[idx, 0, 0, 0]
            for n, idx in zip(obs_sizes, self.indices)
        ])

    def _coarsen_to_observation_grid(self, x: torch.Tensor) -> Tuple[Tensor, ...]:
        r"""Coarsen variables to observation grid resolution.

        Arguments:
            x: Input tensor of shape (C, T, H, W).

        Returns:
            Tuple of coarsened tensors.
        """
        # Unscaling and extracting surface values
        x = self.unscale(x)
        x = x[self.indices, 0, :, :]

        # Coarsen each variable to its observation resolution
        coarsened = [
            create_coarsen_variable(
                x[i, :, :],
                lon_src=self.coordinates["longitude"],
                lat_src=self.coordinates["latitude"],
                lon_tgt=self.coarsened_coordinates[var]["longitude"],
                lat_tgt=self.coarsened_coordinates[var]["latitude"],
                target_resolution=OBSERVATIONS_RESOLUTION[var.lower()],
            )
            for i, var in enumerate(self.variable_names)
        ]

        return tuple(coarsened)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Applies the observation operator.

        Arguments:
            x: Input tensor of shape (C, T, H, W).

        Returns:
            A(x) + mu_y
        """
        # Coarsen to observation grid
        coarsened_vars = self._coarsen_to_observation_grid(x)

        # Extract values at observed locations using masks
        observations = [
            coarsened[torch.tensor(self.coarsened_masks[var], device=DEVICE) == 1]
            for coarsened, var in zip(coarsened_vars, self.variable_names)
        ]

        # Concatenate all observations
        y = torch.concat(observations, dim=0)

        # Add physical bias if provided
        if self.mu_y is not None:
            y = y + self.mu_y

        # Scale to standardized space
        return self.scale(y)

    def visualize(self, x: torch.Tensor) -> Tuple[Tensor, ...]:
        r"""Visualizes observation operator effect.

        Arguments:
            x: Input tensor of shape (C, T, H, W).

        Returns:
            Tuple of coarsened tensors with NaN where no observations.
        """
        # Coarsen to observation grid
        coarsened_vars = self._coarsen_to_observation_grid(x)

        # Set to NaN where mask indicates no observation
        for coarsened, var in zip(coarsened_vars, self.variable_names):
            coarsened[torch.tensor(self.coarsened_masks[var], device=DEVICE) == 0] = np.nan

        return tuple(coarsened_vars)

    def unscale(self, x: Tensor) -> Tensor:
        r"""Unstandardizes the input tensor."""
        return x * self.sigma_x + self.mu_x if self.unscaling else x

    def scale(self, x: Tensor) -> Tensor:
        r"""Standardizes input tensor."""
        return (x - self.mu_x_obs) / self.sigma_x_obs if self.unscaling else x

    def get_observation_statistics(self) -> Tuple[Tensor, Tensor]:
        r"""Returns training dataset mean and standard deviation at observed locations."""
        return self.mu_x_obs, self.sigma_x_obs
