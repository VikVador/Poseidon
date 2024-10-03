r"""Diffusion - Custom weighted loss function."""

import torch

# isort: split
from poseidon.diffusion.denoiser import PoseidonDenoiser


def PoseidonLoss(
    denoiser: PoseidonDenoiser, x: torch.Tensor, sigma_t: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    r"""Computes a weighted loss for the denoising model which emphasizes
        the error differently based on the noise level.

    References:
        | Elucidating the Design Space of Diffusion-Based Generative Models (Karras et al., 2022).
        | https://arxiv.org/abs/2206.00364

    Arguments:
        denoiser: Denoising model to predict the clean data from noisy data.
        x: Original input tensor with shape (B, D).
        sigma_t: Noises scale applied to the input (B, 1).
        c:  Tokenized time-based conditioning (B, 3).

    Returns:
        Weighted mean squared error loss (scalar tensor).
    """
    x_t = x + sigma_t * torch.randn_like(x)
    x_hat = denoiser(x_t=x_t, sigma_t=sigma_t, c=c)
    error = x_hat - x
    lambda_t = 1 + 1 / (sigma_t**2)
    return torch.mean(lambda_t * torch.mean(error**2, dim=-1))
