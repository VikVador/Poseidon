r"""Diffusion loss."""

import torch

from einops import rearrange
from torch import Tensor, nn

# isort: split
from poseidon.data.mask import generate_trajectory_mask

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def get_loss_level_weights(mask: Tensor) -> Tensor:
    r"""Computes level weights using inverse sea spatial dimensions fraction."""

    pixels_total = mask.sum(dim=(1, 3, 4))[0, 0]
    pixels_per_layer = mask.sum(dim=(3, 4))[0, :, 0] / pixels_total

    # Min-max normalization
    norm = (pixels_per_layer - pixels_per_layer.min()) / (
        pixels_per_layer.max() - pixels_per_layer.min()
    )

    # Inverting weights
    weights_per_layer = 1 + (1 - norm)
    if torch.isnan(weights_per_layer).any():
        weights_per_layer = torch.ones_like(weights_per_layer)
    weights_per_layer = weights_per_layer[None, :, None, None, None]

    # Expanding to match state shape
    _, _, K, X, Y = mask.shape
    return weights_per_layer.expand(-1, -1, K, X, Y)


class PoseidonLoss(nn.Module):
    r"""Masked weighted (level & noise) loss.

    References:
        | Elucidating the Design Space of Diffusion-Based Generative Models (Karras et al., 2022).
        | https://arxiv.org/abs/2206.00364

    Arguments:
        blanket_size: Total number of elements in a blanket (K).
    """

    def __init__(self, blanket_size: int):
        super().__init__()

        self.mask = rearrange(
            generate_trajectory_mask(trajectory_size=blanket_size), "B C K X Y -> B (C K X Y)"
        ).to(DEVICE)

        self.weight_levels = rearrange(
            get_loss_level_weights(mask=generate_trajectory_mask(trajectory_size=blanket_size)),
            "B C K X Y -> B (C K X Y)",
        ).to(DEVICE)

        self.weight_levels = self.weight_levels[:, self.mask[0] == 1]

    def forward(self, x_0: Tensor, x_0_denoised: Tensor, sigma_t: Tensor) -> Tensor:
        r"""
        Arguments:
            x_0: Ground truth (B, C * K * X * Y).
            x_0_denoised: Clean tensor estimate (B, C * K * X * Y).
            sigma_t: Associated noise levels (B, 1).
        """

        x_0, x_0_denoised = (x_0[:, self.mask[0] == 1], x_0_denoised[:, self.mask[0] == 1])

        # Level-wise weighted SE
        se = self.weight_levels * (x_0_denoised - x_0) ** 2

        # Noise level weigth error
        weight_noise = 1 + 1 / (sigma_t**2)
        mse = torch.mean(se, dim=-1, keepdim=True)
        wmse = weight_noise * mse
        mwmse = torch.mean(wmse)

        return mwmse
