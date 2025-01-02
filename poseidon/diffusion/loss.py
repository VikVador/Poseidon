r"""Diffusion training loss."""

import torch

from torch import Tensor


def PoseidonLoss(
    x_t: Tensor,
    x_t_denoised: Tensor,
    sigma_t: Tensor,
) -> Tensor:
    r"""Weighted loss which emphasizes the error differently based on the noise level.

    References:
        | Elucidating the Design Space of Diffusion-Based Generative Models (Karras et al., 2022).
        | https://arxiv.org/abs/2206.00364

    Arguments:
        x_t: Noisy tensor (B, D).
        x_t_denoised: Denoised tensor (B, D).
        sigma_t: Noises scale originally applied to input (B, 1).

    Returns:
        Weighted mean squared error (scalar tensor).
    """
    # fmt: off
    #
    # Sigma should never be zero due to exponential scaling.
    lambda_t = 1 / (1 + sigma_t**2)

    se       = (x_t_denoised - x_t) ** 2
    mse      = torch.mean(se, dim=-1)
    wmse     = lambda_t * mse
    mwmse    = torch.mean(wmse)
    return mwmse
