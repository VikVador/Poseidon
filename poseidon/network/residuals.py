r"""Modulated Residual Convolutional blocks.

Inspired by: https://github.com/probabilists/azula

"""

import torch
import torch.nn as nn

from torch import Tensor
from typing import Optional

# isort: split
from poseidon.network.convolutions import ConvNd
from poseidon.network.modulation import Modulator
from poseidon.network.normalization import LayerNorm
from poseidon.network.tools import reshape, unshape


class ModulatedResidualBlock(nn.Module):
    r"""Base class for a modulated residual convolutional block.

    Arguments:
        channels: Number of channels.
        mod_features: Number of features (D) in the modulating vector (B, D).
        spatial: Number of spatial dimensions (used for convolution) in the target signal.
        dropout: Dropout rate [0, 1].
        kwargs: Keyword arguments passed to :class:`torch.nn.Conv1d` or :class:`torch.nn.Conv2d`.
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

        self.convolution_type = "temporal" if spatial == 1 else "spatial"

        self.ada_zero = Modulator(
            channels=channels,
            mod_features=mod_features,
            spatial=spatial,
        )

        self.block = nn.Sequential(
            LayerNorm(dim=1),
            ConvNd(
                in_channels=channels,
                out_channels=channels,
                spatial=spatial,
                **kwargs,
            ),
            nn.SiLU(),
            nn.Identity() if dropout is None else nn.Dropout(dropout),
            ConvNd(
                in_channels=channels,
                out_channels=channels,
                spatial=spatial,
                **kwargs,
            ),
        )

    def forward(self, x: Tensor, mod: Tensor) -> Tensor:
        r"""Computes the output of the modulated residual convolutional block.

        Arguments:
            x: Input tensor, with shape (B, C, T, H, W).
            mod: Modulation vector, with shape (B, D).
        """

        # Hidding dimension(s)
        x, mod, original_shape = reshape(
            convolution=self.convolution_type,
            x=x,
            mod=mod,
        )

        # Modulated Residual Convolutional Block
        a, b, c = self.ada_zero(mod)
        y = (a + 1) * x + b
        y = self.block(y)
        y = x + c * y
        y = y / torch.sqrt(1 + c * c)

        # Restoring dimension(s)
        y = unshape(
            convolution=self.convolution_type,
            x=x,
            shape=original_shape,
        )

        return y


class SpatialModulatedResidualBlock(ModulatedResidualBlock):
    r"""Creates a modulated residual 2D convolutional block, independent of the temporal dimension."""

    def __init__(
        self,
        channels: int,
        mod_features: int,
        dropout: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(channels, mod_features, spatial=2, dropout=dropout, **kwargs)


class TemporalModulatedResidualBlock(ModulatedResidualBlock):
    r"""Creates a modulated residual 1D convolutional block, independent of the spatial dimensions."""

    def __init__(
        self,
        channels: int,
        mod_features: int,
        dropout: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(channels, mod_features, spatial=1, dropout=dropout, **kwargs)
