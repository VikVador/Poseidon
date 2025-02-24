r"""Diffusion training loss."""

import torch

from einops import rearrange
from torch import Tensor, nn
from typing import Dict, Sequence, Tuple

# isort: split
from poseidon.diffusion.tools import get_mask_variables

# Constants
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class PoseidonLoss(nn.Module):
    r"""Weighted loss which emphasizes the error differently based on the noise level.

    References:
        | Elucidating the Design Space of Diffusion-Based Generative Models (Karras et al., 2022).
        | https://arxiv.org/abs/2206.00364

    Arguments:
        variables: Variable names to retain from the preprocessed dataset.
        region: Region of interest to extract from the dataset.
        blanket_size: Total number of elements in a blanket.
    """

    def __init__(
        self,
        variables: Sequence[str],
        region: Dict[str, Tuple[int, int]],
        blanket_size: int,
    ):
        super().__init__()

        self.mask = rearrange(
            get_mask_variables(
                variables=variables,
                region=region,
                blanket_size=blanket_size,
            ),
            "B ... -> B (...)",
        ).to(DEVICE)

    def forward(
        self,
        x: Tensor,
        x_denoised: Tensor,
        sigma: Tensor,
    ) -> Tensor:
        r"""
        Arguments:
            x: Clean tensor (B, D).
            x_denoised: Denoised tensor (B, D).
            sigma: Noise scale applied on x (B, 1).

        Returns:
            Mean weighted mean squared error.
        """
        # fmt: off
        #
        # Extracting sea values
        x, x_denoised = (
            x         [:, self.mask[0] == 1],
            x_denoised[:, self.mask[0] == 1],
        )

        weight   = (1 / sigma ** 2) + 1
        se       = (x_denoised - x) ** 2
        mse      = torch.mean(se, dim=-1, keepdim=True)
        wmse     = weight * mse
        mwmse    = torch.mean(wmse)

        return mwmse
