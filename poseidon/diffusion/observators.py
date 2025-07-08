r"""Observation operators."""

import numpy as np
import torch
import torch.nn as nn

from abc import abstractmethod
from torch import Tensor
from typing import Dict, Sequence, Tuple

# isort: split
from poseidon.config import PATH_MASK_O
from poseidon.data.mask import generate_trajectory_mask

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class A(nn.Module):
    r"""A template to create masks.

    Arguments:
        variables: List of variable names to extract.
        region: Dictionary specifying region slicing.
    """

    def __init__(
        self,
        variables: Sequence[str],
        region: Dict[str, Tuple[int, int]],
    ):
        super().__init__()

        # Creating mask for nowcasts
        self.mask = generate_trajectory_mask(
            variables=variables,
            region=region,
            trajectory_size=1,
        )[0]

        # Indexes of surface elements
        self.surface_indexes = [32, 64, 96, 128]

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        r"""Extracts the values using mask"""

    @abstractmethod
    def visualize(self, x: Tensor) -> Tensor:
        r"""Visualizes the mask applied to the input tensor."""


class A_surface(A):
    """Observe completely the surface."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts the surface values using the mask."""

        # Extracting the surfaces
        x_, mask_ = (
            x[self.surface_indexes, 0, :, :],
            self.mask[self.surface_indexes, 0, :, :].to(x.device),
        )

        # Extracting values
        return x_[mask_ == 1]

    def visualize(self, x: torch.Tensor) -> torch.Tensor:
        """Visualizes the mask applied to the input tensor."""

        # Extracting the surfaces
        x_, mask_ = (
            x[self.surface_indexes, 0, :, :],
            self.mask[self.surface_indexes, 0, :, :].to(x.device),
        )

        # Masking the land
        x_[mask_ == 0] = np.nan

        # Returning the visualization
        return x_


class A_coarsen(A):
    """Observe surface at lower resolution."""

    def __init__(
        self, coarsening_factor: int, variables: Sequence[str], region: Dict[str, Tuple[int, int]]
    ):
        super().__init__(variables=variables, region=region)

        # Factor by which x and y dimensions are divided
        self.coarsening_factor = coarsening_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts the surface values using the mask."""

        # Extracting the surfaces
        x_, mask_ = (
            x[self.surface_indexes, 0, :, :],
            self.mask[self.surface_indexes, 0, :, :].to(x.device),
        )

        # Coarsening
        x_, mask_ = (
            x_[:, :: self.coarsening_factor, :: self.coarsening_factor],
            mask_[:, :: self.coarsening_factor, :: self.coarsening_factor],
        )

        # Extracting values
        return x_[mask_ == 1]

    def visualize(self, x: torch.Tensor) -> torch.Tensor:
        """Visualizes the mask applied to the input tensor."""

        # Extracting the surfaces
        x_, mask_ = (
            x[self.surface_indexes, 0, :, :],
            self.mask[self.surface_indexes, 0, :, :].to(x.device),
        )

        # Coarsening
        x_, mask_ = (
            x_[:, :: self.coarsening_factor, :: self.coarsening_factor],
            mask_[:, :: self.coarsening_factor, :: self.coarsening_factor],
        )

        # Masking the land
        x_[mask_ == 0] = np.nan

        # Returning the visualization
        return x_


class A_partial(A):
    """Observe partially the surface."""

    def __init__(self, variables: Sequence[str], region: Dict[str, Tuple[int, int]]):
        super().__init__(variables=variables, region=region)

        self.mask_partial = (
            torch.stack([torch.load(p, weights_only=False) for p in PATH_MASK_O]) * 1.0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts the surface values using the mask."""

        # Extracting the surfaces
        x_, mask_ = (x[self.surface_indexes, 0, :, :], self.mask[self.surface_indexes, 0, :, :])

        # Applying the partial mask to current mask
        mask_[self.mask_partial == 0] = 0

        # Extracting values
        return x_[mask_ == 1]

    def visualize(self, x: torch.Tensor) -> torch.Tensor:
        """Visualizes the mask applied to the input tensor."""

        # Extracting the surfaces
        x_, mask_ = (
            x[self.surface_indexes, 0, :, :],
            self.mask[self.surface_indexes, 0, :, :].to(x.device),
        )

        # Applying the partial mask to current mask
        mask_[self.mask_partial == 0] = 0

        # Masking the land
        x_[mask_ == 0] = np.nan

        # Returning the visualization
        return x_
