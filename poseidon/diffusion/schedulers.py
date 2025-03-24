r"""Diffusion schedulers."""

import math
import torch
import torch.nn as nn

from torch import Tensor


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
    r"""Creates a log-logit noise schedule.

    Mathematics:
        sigma_t = sqrt(sigma_min * sigma_max) * exp(rho * logit(t))

    Arguments:
        sigma_min: Initial noise scale.
        sigma_max: Final noise scale.
        spread: Spread factor for the noise scale.

    Returns
        Tensor (B, 1).
    """

    def __init__(self, sigma_min: float = 1e-5, sigma_max: float = 1e3, spread: float = 2.0):
        super().__init__()

        self.spread = spread
        self.eps = math.sqrt(sigma_min / sigma_max) ** (1 / spread)
        self.log_sigma_min = math.log(sigma_min)
        self.log_sigma_max = math.log(sigma_max)
        self.log_sigma_med = math.log(sigma_min * sigma_max) / 2

    def forward(self, t: Tensor) -> Tensor:
        return torch.exp(
            self.spread * torch.logit(t * (1 - 2 * self.eps) + self.eps) + self.log_sigma_med
        )
