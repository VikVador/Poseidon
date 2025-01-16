r"""Diffusion training loss."""

import torch

from torch import Tensor


def PoseidonLoss(
    x: Tensor,
    x_denoised: Tensor,
    sigma: Tensor,
) -> Tensor:
    r"""Weighted loss which emphasizes the error differently based on the noise level.

    References:
        | Elucidating the Design Space of Diffusion-Based Generative Models (Karras et al., 2022).
        | https://arxiv.org/abs/2206.00364

    Arguments:
        x: Clean tensor (B, D).
        x_denoised: Denoised tensor (B, D).
        sigma: Noise scale applied on x (B, 1).

    Returns:
        Mean weighted mean squared error.
    """
    # fmt: off
    #
    weight   = (1 / sigma**2) + 1

    se       = (x_denoised - x) ** 2
    mse      = torch.mean(se, dim=-1, keepdim=True)
    wmse     = weight * mse
    mwmse    = torch.mean(wmse)
    return mwmse
