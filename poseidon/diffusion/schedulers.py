r"""Diffusion schedulers."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Sequence

# isort: split
from poseidon.diffusion.const import DATASET_COV_SQRT_EIGEN_VALUES


class PoseidonTimeScheduler(nn.Module):
    r"""A custom time scheduler for diffusion models.

    Mathematics:
        t ~ U(0, 1)

    Information:
        This scheduler samples timesteps uniformly from [0, 1]
        to be used in the diffusion process. Biasing the
        sampling of timesteps does not appear to
        improve model training.
    """

    def __init__(self):
        super().__init__()

    def forward(self, batch_size: int) -> Tensor:
        r"""Generates a batch of random timesteps.

        Arguments:
            batch_size: Number of random timesteps to generate.

        Returns:
            Tensor (B, 1).
        """
        return torch.rand(batch_size)

    def get_timesteps(self, index: int, steps: int, batch_size: int) -> Tensor:
        r"""Computes a fixed timestep value for a given index.

        Arguments:
            index: Position in the linearly discretized time axis.
            steps: Total number of time steps.
            batch_size: Number of times to replicate the computed timestep.

        Returns:
            Tensor (B, 1).
        """
        assert (
            index < steps
        ), "ERROR (PoseidonTimeScheduler) - Index must be less than the number of steps."
        return torch.full((batch_size,), index / steps)


class PoseidonNoiseScheduler(nn.Module):
    r"""A custom noise scheduler for diffusion models.

    Information:
        This scheduler determines noise levels (`sigma`) based on a predefined
        sequence of values, using piece-wise linear interpolation.
    """

    def __init__(self, noise_levels: Sequence[float] = DATASET_COV_SQRT_EIGEN_VALUES):
        super().__init__()

        self.noise_levels = torch.tensor(noise_levels, dtype=torch.float32).view(1, 1, -1)
        self.num_levels = len(noise_levels)

    def forward(self, timesteps: Tensor) -> Tensor:
        r"""Computes interpolated noise levels (`sigma`) for given timesteps.

        Arguments:
            timesteps: Tensor of shape (B, 1), where each value is a timestep in [0,1].

        Returns:
            A Tensor of shape (B, 1) containing the interpolated noise levels.
        """

        assert (0 <= timesteps).all() and (
            timesteps <= 1
        ).all(), "ERROR (PoseidonNoiseScheduler) - Timesteps must be in range [0, 1]."

        # Compute scaled timesteps for interpolation
        scaled_t = (timesteps * (self.num_levels - 1)).view(1, -1, 1)

        # Determine noise levels using interpolation
        return (
            F.interpolate(
                self.noise_levels,
                scale_factor=scaled_t.shape[1] / self.num_levels,
                mode="linear",
                align_corners=True,
            )
            .squeeze(0)
            .view(-1, 1)
        )
