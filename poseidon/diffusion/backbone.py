r"""Diffusion backbone."""

import torch
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
from poseidon.network.udit import UDiT


class PoseidonBackbone(nn.Module):
    r"""Helper module to preprocess data before denoising.

    Arguments:
        variables: Variable names to retain from the dataset.
        dimensions: Input tensor dimensions (B, C, K, X, Y).
        config_unet: Configuration of the unet.
        config_siren: Configuration of the siren architecture.
        config_region: Configuration of the spatial region.
        config_transformer: Configuration of the transformer.
    """

    def __init__(
        self,
        variables: Sequence[str],
        dimensions: Tuple[int, int, int, int, int],
        config_unet: Dict,
        config_siren: Dict,
        config_region: Dict,
        config_transformer: Dict,
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
            ).bool(),
        )

        self.network = UDiT(
            in_channels=self.C,
            out_channels=self.C,
            config_siren=config_siren,
            config_region=config_region,
            config_transformer=config_transformer,
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

        # Reshaping for network
        x_t = rearrange(
            x_t,
            "B (C K X Y) -> B C K X Y",
            C=self.C,
            K=self.K,
            X=self.X,
            Y=self.Y,
        )

        # Masking land
        x_t = torch.where(self.mask.expand_as(x_t), x_t, LAND_VALUE)

        # Estimating (unscaled) clean signal
        x_t = rearrange(
            self.network(x_t, sigma_t),
            "B C K X Y -> B (C K X Y)",
            C=self.C,
            K=self.K,
            X=self.X,
            Y=self.Y,
        )

        return x_t
