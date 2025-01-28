r"""Diffusion noise scheduler."""

import torch
import torch.nn as nn

from torch import Tensor


class PoseidonNoiseSchedule(nn.Module):
    r"""Log-normal noise schedule.

    Formulation:
        log(sigma) ~ N(mu, sigma^2)

    References:
        | Elucidating the Design Space of Diffusion-Based Generative Models (Karras et al., 2022).
        | https://arxiv.org/abs/2206.00364

    Arguments:
        mu: Mean of the noise in log-space.
        sigma: Standard deviation of the noise in log-space.
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
