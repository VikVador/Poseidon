r"""Diffusion - Custom noise scheduler."""

import torch
import torch.nn as nn


class PoseidonNoiseSchedule(nn.Module):
    r"""Log-Normal Noise Schedule for Diffusion Models.

    Formulation:
        log(sigma) ~ N(mu, sigma^2)

    References:
        | Elucidating the Design Space of Diffusion-Based Generative Models (Karras et al., 2022).
        | https://arxiv.org/abs/2206.00364

    Arguments:
        mu: Mean of the noise in log-space.
        sigma: Standard deviation of the noise in log-space.
    """

    def __init__(self, mu: float = -1.2, sigma: float = 1.2):
        super().__init__()
        self.register_buffer("mu", torch.as_tensor(mu))
        self.register_buffer("sigma", torch.as_tensor(sigma))

    def forward(self, size: int) -> torch.Tensor:
        r"""Generates a given number of noise levels.

        Arguments:
            size: Number of noise levels.

        Returns:
            Noise levels for each sample (size, 1)
        """
        return torch.exp(torch.normal(self.mu, self.sigma, size=(size, 1)))
