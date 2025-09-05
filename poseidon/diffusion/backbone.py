r"""Diffusion backbone."""

import torch
import torch.nn as nn

from einops import rearrange
from torch import Tensor
from typing import (
    Dict,
    Tuple,
)

# isort: split
from poseidon.data.const import LAND_VALUE
from poseidon.data.mask import generate_trajectory_mask
from poseidon.network.unet import UNet


class PoseidonBackbone(nn.Module):
    r"""Helper module for denoising.

    Arguments:
        config_nn: Configuration of neural network.
        dimensions: Input tensor dimensions (B, C, K, X, Y).
    """

    def __init__(
        self,
        config_nn: Dict,
        dimensions: Tuple[int, int, int, int, int],
    ):
        super().__init__()

        self.B, self.C, self.K, self.X, self.Y = dimensions

        self.register_buffer("mask", generate_trajectory_mask(trajectory_size=self.K).bool())

        self.network = UNet(
            in_channels=self.C,
            out_channels=self.C,
            cond_channels=self.K,
            **config_nn,
        )

    def forward(
        self,
        x_t: Tensor,
        sigma_t: Tensor,
        cond: Tensor,
    ) -> Tensor:
        r"""
        Arguments:
            x_t: Noisy tensor (B, C * K * X * Y).
            sigma_t: Associated noise levels (B, 1).
            cond: Associated conditioning tensor (B, K).

        Returns:
            output: Cleaned tensor (B, C * K * X * Y).
        """
        x_t = rearrange(
            x_t,
            "B (C K X Y) -> B C K X Y",
            C=self.C,
            K=self.K,
            X=self.X,
            Y=self.Y,
        )

        x_t = torch.where(self.mask.expand_as(x_t), x_t, LAND_VALUE)  # TO BE CHECKED

        x_t = self.network(x=x_t, mod=sigma_t, cond=cond)

        return rearrange(
            x_t,
            "B C K X Y -> B (C K X Y)",
            C=self.C,
            K=self.K,
            X=self.X,
            Y=self.Y,
        )
