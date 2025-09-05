r"""Observation operators."""

import numpy as np
import torch
import torch.nn as nn

from abc import abstractmethod
from torch import Tensor

# isort: split
from poseidon.data.mask import generate_trajectory_mask

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class A(nn.Module):
    r"""An observator template."""

    def __init__(self):
        super().__init__()

        self.mask = generate_trajectory_mask(trajectory_size=1)[0]

        # Indexes of surface elements (To Be Generalized)
        self.surface_indexes = [32, 64, 96, 128]

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        r"""Extracts values using mask"""

    @abstractmethod
    def visualize(self, x: Tensor) -> Tensor:
        r"""Visualizes mask effect."""


class A_surface(A):
    """Observe completely the surface."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Extracts values using mask"""

        x_, mask_ = (
            x[self.surface_indexes, 0, :, :],
            self.mask[self.surface_indexes, 0, :, :].to(x.device),
        )

        return x_[mask_ == 1]

    def visualize(self, x: torch.Tensor) -> torch.Tensor:
        r"""Visualizes mask effect."""

        x_, mask_ = (
            x[self.surface_indexes, 0, :, :],
            self.mask[self.surface_indexes, 0, :, :].to(x.device),
        )

        x_[mask_ == 0] = np.nan

        return x_


class A_coarsen(A):
    """Observe surface at lower resolution."""

    def __init__(self, coarsening_factor: int):
        super().__init__()
        self.coarsening_factor = coarsening_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Extracts values using mask"""

        x_, mask_ = (
            x[self.surface_indexes, 0, :, :],
            self.mask[self.surface_indexes, 0, :, :].to(x.device),
        )

        x_, mask_ = (
            x_[:, :: self.coarsening_factor, :: self.coarsening_factor],
            mask_[:, :: self.coarsening_factor, :: self.coarsening_factor],
        )

        return x_[mask_ == 1]

    def visualize(self, x: torch.Tensor) -> torch.Tensor:
        r"""Visualizes mask effect."""

        x_, mask_ = (
            x[self.surface_indexes, 0, :, :],
            self.mask[self.surface_indexes, 0, :, :].to(x.device),
        )

        x_, mask_ = (
            x_[:, :: self.coarsening_factor, :: self.coarsening_factor],
            mask_[:, :: self.coarsening_factor, :: self.coarsening_factor],
        )

        x_[mask_ == 0] = np.nan

        return x_
