r"""Diffusion noise scheduler."""

import torch
import torch.nn as nn

from torch import Tensor


class PoseidonNormalLogNoiseSchedule(nn.Module):
    r"""Normal log-noise schedule.

    Formulation:
        log(sigma) ~ N(mu, sigma^2)

    References:
        | Elucidating the Design Space of Diffusion-Based Generative Models (Karras et al., 2022).
        | https://arxiv.org/abs/2206.00364

    Arguments:
        mu: Mean of log(sigma).
        sigma: Standard deviation of log(sigma).
    """

    def __init__(self, mu: float = -1.0, sigma: float = 1.7):
        super().__init__()

        self.register_buffer("mu", torch.as_tensor(mu))
        self.register_buffer("sigma", torch.as_tensor(sigma))

    def forward(self, batch_size: int) -> Tensor:
        r"""Generates a batch of noise levels.

        Arguments:
            batch_size: Batch size (B) of input tensor.

        Returns:
            Tensor: Noise levels (B, 1)
        """
        return torch.exp(
            torch.normal(
                self.mu,
                self.sigma,
                size=(batch_size, 1),
            )
        ).to(dtype=torch.float32)


class PoseidonUniformLogNoiseSchedule(nn.Module):
    r"""Uniform log-noise schedule.

    Formulation:
        log(sigma) ~ Uniform(log(sigma_min), log(sigma_max))

    References:
        | Elucidating the Design Space of Diffusion-Based Generative Models (Karras et al., 2022).
        | https://arxiv.org/abs/2206.00364

    Arguments:
        sigma_min: Minimum value of noise.
        sigma_max: Maximum value of noise.
    """

    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 100.0):
        super().__init__()

        self.register_buffer(
            "log_sigma_min",
            torch.log(torch.tensor(sigma_min)),
        )
        self.register_buffer(
            "log_sigma_max",
            torch.log(torch.tensor(sigma_max)),
        )

    def forward(self, batch_size: int) -> Tensor:
        r"""Generates a batch of noise levels.

        Arguments:
            batch_size: Batch size (B) of input tensor.

        Returns:
            Tensor: Noise levels (B, 1)
        """
        return torch.exp(
            (
                torch.rand((batch_size, 1)) * (self.log_sigma_max - self.log_sigma_min)
                + self.log_sigma_min
            )
        ).to(dtype=torch.float32)
