r"""Diffusion schedulers."""

import torch
import torch.nn as nn

from torch import Tensor
from typing import Sequence

# fmt: off
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
        return torch.rand(batch_size).unsqueeze(1)


class PoseidonNoiseScheduler(nn.Module):
    r"""A custom noise scheduler for diffusion models.

    Information:
        This scheduler determines noise levels (`sigma`) based on a predefined
        sequence of values, using piece-wise linear interpolation.
    """

    def __init__(self, noise_levels: Sequence[float] = DATASET_COV_SQRT_EIGEN_VALUES):
        super().__init__()
        self.register_buffer("noise_levels", torch.tensor(noise_levels, dtype=torch.float32))

    def forward(self, timesteps: Tensor) -> Tensor:
        r"""Computes interpolated noise levels (`sigma`) for given timesteps.

        Arguments:
            timesteps: Tensor of shape (B, 1), where each value is a timestep in [0,1].

        Returns:
            A Tensor of shape (B, 1) containing the interpolated noise levels.
        """
        assert (
            timesteps.ndim == 2 and timesteps.shape[1] == 1
        ), "ERROR (PoseidonNoiseScheduler) - Timesteps must have shape (B, 1)"

        N            = self.noise_levels.shape[0]
        indices      = timesteps * (N - 1)
        lower_idx    = torch.floor(indices).long()
        upper_idx    = torch.ceil(indices).long().clamp(max=N - 1)
        lower_values = self.noise_levels[lower_idx]
        upper_values = self.noise_levels[upper_idx]
        weights      = indices - lower_idx

        return (1 - weights) * lower_values + weights * upper_values
