r"""Modulated Residual Convolutional blocks."""

import torch
import torch.nn as nn

from torch import Tensor
from typing import Optional

# isort: split
from poseidon.network.convolution import ConvNd
from poseidon.network.modulation import Modulator
from poseidon.network.normalization import LayerNorm
from poseidon.network.tools import reshape, unshape


class ModulatedResidualBlock(nn.Module):
    r"""Base class for a modulated residual convolutional block.

    Information:
        The input tensor has shape (B, C, T, H, W), where T is the temporal dimension,
        and H, W are the spatial dimensions.

    Spatial:
        If `spatial=1`, the convolution is applied along the temporal dimension (T).
        If `spatial=2`, the convolution is applied along the spatial dimensions (H, W).

    Arguments:
        channels: Number of input channels (C).
        mod_features: Number of features (D) in the modulating vector (B, D).
        spatial: Number of spatial dimensions on which the convolution is applied.
        dropout: Dropout rate [0, 1].
        kwargs: Keyword arguments passed to :class:`torch.nn.ConvXd`.
    """

    def __init__(
        self,
        channels: int,
        mod_features: int,
        spatial: int,
        dropout: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()

        self.hidden_dimension = "space" if spatial == 1 else "time"

        self.modulator = Modulator(
            channels=channels,
            mod_features=mod_features,
            spatial=spatial,
        )

        self.convolution_block = nn.Sequential(
            LayerNorm(dim=1),
            ConvNd(
                in_channels=channels,
                out_channels=channels,
                spatial=spatial,
                **kwargs,
            ),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout is not None else nn.Identity(),
            ConvNd(
                in_channels=channels,
                out_channels=channels,
                spatial=spatial,
                **kwargs,
            ),
        )

    def forward(self, x: Tensor, mod: Tensor) -> Tensor:
        r"""
        Arguments:
            x: Input tensor (B, C, T, H, W).
            mod: Modulation vector (B, D).

        Returns:
            Tensor: Output tensor (B, C, T, H, W).
        """
        x, mod, original_shape = reshape(hide=self.hidden_dimension, x=x, mod=mod)

        mod_factor, mod_bias, mod_scaling = self.modulator(mod)
        y = (mod_factor + 1) * x + mod_bias
        y = self.convolution_block(y)
        y = x + mod_scaling * y
        y = y / torch.sqrt(1 + mod_scaling * mod_scaling)
        return unshape(
            extract=self.hidden_dimension,
            x=x,
            shape=original_shape,
        )


class SpatialModulatedResidualBlock(ModulatedResidualBlock):
    r"""A residual convolutional block with 2D spatial modulation."""

    def __init__(
        self,
        channels: int,
        mod_features: int,
        dropout: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(channels, mod_features, spatial=2, dropout=dropout, **kwargs)


class TemporalModulatedResidualBlock(ModulatedResidualBlock):
    r"""A residual convolutional block with 1D temporal modulation."""

    def __init__(
        self,
        channels: int,
        mod_features: int,
        dropout: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(channels, mod_features, spatial=1, dropout=dropout, **kwargs)
