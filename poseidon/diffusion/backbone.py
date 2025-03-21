r"""Diffusion backbone."""

import torch.nn as nn

from einops import rearrange
from torch import Tensor
from typing import (
    Dict,
    Sequence,
    Tuple,
)

# isort: split
from poseidon.data.const import LAND_VALUE
from poseidon.data.mask import generate_trajectory_mask
from poseidon.network.unet import UNet


class PoseidonBackbone(nn.Module):
    r"""Helper module used to add conditionning before denoising.

    References:
        | Elucidating the Design Space of Diffusion-Based Generative Models (Karras et al., 2022).
        | https://arxiv.org/abs/2206.00364

    Arguments:
        variables: Variable names to retain from the dataset.
        dimensions: Input tensor dimensions (B, C, K, X, Y).
        config_unet: Configuration for UNet architecture.
        config_siren: Configuration for the Siren architecture.
        config_region: Configuration for the spatial region.
    """

    def __init__(
        self,
        variables: Sequence[str],
        dimensions: Tuple[int, int, int, int, int],
        config_unet: Dict,
        config_siren: Dict,
        config_region: Dict,
    ):
        super().__init__()

        self.B, self.C, self.K, self.X, self.Y = dimensions

        # Land & Sea mask
        self.register_buffer(
            "mask",
            generate_trajectory_mask(
                variables=variables,
                region=config_region,
                trajectory_size=self.K,
            ),
        )

        # Total number of channels
        channels = self.C * self.K

        # 2D UNet
        self.unet = UNet(
            in_channels=channels,
            out_channels=channels,
            config_region=config_region,
            config_siren=config_siren,
            **config_unet,
        )

    def forward(
        self,
        x_t: Tensor,
        sigma_t: Tensor,
    ) -> Tensor:
        r"""Denoising conditionned sample.

        Arguments:
            x_t: Noisy input tensor (B, C * K * X * Y).
            sigma_t: Associated noise levels (B, 1).

        Returns:
            Cleaned tensor (B, C * K * X * Y).
        """

        x_t = rearrange(
            x_t,
            "B (C K X Y) -> B C K X Y",
            C=self.C,
            K=self.K,
            X=self.X,
            Y=self.Y,
        )

        # Masking land
        x_t[:, self.mask[0] == 0] = LAND_VALUE

        # Merging time and levels into channels (UNet 2D)
        x_t = rearrange(x_t, "B C K X Y -> B (C K) X Y")

        # Estimating (unscaled) clean signal
        return rearrange(
            self.unet(x_t, sigma_t),
            "B (C K) X Y -> B (C K X Y)",
            C=self.C,
            K=self.K,
            X=self.X,
            Y=self.Y,
        )
