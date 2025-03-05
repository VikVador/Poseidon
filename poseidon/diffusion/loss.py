r"""Diffusion loss."""

import torch

from einops import rearrange
from torch import Tensor, nn
from typing import Dict, Sequence, Tuple

# isort: split
from poseidon.data.mask import generate_trajectory_mask

# fmt: off
#
# Constants
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class PoseidonLoss(nn.Module):
    r"""Weighted loss (masked) emphasizing the error differently based on the noise level.

    References:
        | Elucidating the Design Space of Diffusion-Based Generative Models (Karras et al., 2022).
        | https://arxiv.org/abs/2206.00364

    Arguments:
        variables: Variable names to retain from the dataset.
        region: Region of interest to extract from the dataset.
        blanket_size: Total number of elements in a blanket (K).
        use_mask: Whether to compute the error only on the sea or not.
    """

    def __init__(
        self,
        variables: Sequence[str],
        region: Dict[str, Tuple[int, int]],
        blanket_size: int,
        use_mask: bool = True,
    ):
        super().__init__()

        self.use_mask, self.mask = (
            use_mask,
            rearrange(
                generate_trajectory_mask(
                    variables=variables,
                    region=region,
                    trajectory_size=blanket_size,
                ),
                "B C K X Y -> B (C K X Y)",
            ).to(DEVICE),
        )

    def forward(
        self,
        x_0: Tensor,
        x_0_denoised: Tensor,
        sigma_t: Tensor,
    ) -> Tensor:
        r"""
        Arguments:
            x_0: Clean tensor (B, C * K * X * Y).
            x_0_denoised: Estimate of clean tensor (B, C * K * X * Y).
            sigma_t: Associated noise levels (B, 1).
        """
        if self.use_mask:
            x_0, x_0_denoised = (
                x_0         [:, self.mask[0] == 1],
                x_0_denoised[:, self.mask[0] == 1],
            )

        weight   = (1 / sigma_t ** 2) + 1
        se       = (x_0_denoised - x_0) ** 2
        mse      = torch.mean(se, dim=-1, keepdim=True)
        wmse     = weight * mse
        mwmse    = torch.mean(wmse)

        return mwmse
