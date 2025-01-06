r"""UNet Architecture for 3-dimensional convolutions.

This UNet performs 3D convolutions via a combination of 2D spatial
and 1D temporal convolutions, optimizing efficiency for diffusion tasks.

References:
    Inspired by the implementation in:
    https://github.com/probabilists/azula
"""

import torch
import torch.nn as nn

from torch import Tensor
from typing import Optional, Sequence

# isort: split
from poseidon.network.convolution import ConvNd, Convolution2DBlock
from poseidon.network.normalization import LayerNorm
from poseidon.network.residual import (
    SpatialModulatedResidualBlock,
    TemporalModulatedResidualBlock,
)
from poseidon.network.upsample import UpsampleBlock


class UNetBlock(nn.Module):
    r"""A UNet block combining spatial and temporal residual convolutions.

    Arguments:
        channels: Number of channels (C) in the input tensor.
        mod_features: Number of features (D) in the modulation vector.
        dropout: Dropout probability for regularization [0, 1].
        **kwargs: Additional arguments passed to the residual blocks.
    """

    def __init__(
        self,
        channels: int,
        mod_features: int,
        dropout: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()

        properties = {
            "channels": channels,
            "mod_features": mod_features,
            "dropout": dropout,
        }

        self.block_spatial = SpatialModulatedResidualBlock(
            **properties,
            **kwargs,
        )
        self.block_temporal = TemporalModulatedResidualBlock(
            **properties,
            **kwargs,
        )

    def forward(self, x: Tensor, mod: Tensor) -> Tensor:
        """
        Arguments:
            x: Input tensor with shape (B, C, T, H, W).
            mod: Modulation vector with shape (B, D).

        Returns:
            Tensor: Convoluted tensor (B, C, T, H, W).
        """
        x = self.block_temporal(x, mod)
        x = self.block_spatial(x, mod)
        return x


class UNet(nn.Module):
    r"""Creates a U-Net model for 3-dimensional convolutions.

    Example:
        >>> unet = UNet(in_channels=64,
                        out_channels=64,
                        mod_features=128,
                        hid_channels=[64, 128, 256],
                        hid_blocks=[2, 3, 3],
                        kernel_size=3,
                        stride=2,
                        dropout=0.1)

    Arguments:
        in_channels: Number of input channels (C_i)
        out_channels: Number of output channels (C_o).
        mod_features: Number of features (D) in the modulating vector (B, D).
        hid_channels: Numbers of channels at each depth.
        hid_blocks: Numbers of hidden blocks at each depth.
        kernel_size: Kernel size of all convolutions.
        stride: Stride of the downsampling convolutions.
        dropout: Dropout probability for regularization [0, 1].
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mod_features: int,
        hid_channels: Sequence[int] = (64, 128, 256),
        hid_blocks: Sequence[int] = (2, 3, 3),
        kernel_size: int = 3,
        stride: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()

        assert len(hid_blocks) == len(hid_channels)
        kwargs = dict(
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        # Contains the blocks for the descent and ascent of the UNet
        self.descent, self.ascent = nn.ModuleList(), nn.ModuleList()

        # Creation of the UNet
        for i, blocks in enumerate(hid_blocks):
            # Contains blocks at a stage of the UNet
            do, up = nn.ModuleList(), nn.ModuleList()

            # Filling the stage with blocks
            for _ in range(blocks):
                do.append(
                    UNetBlock(
                        hid_channels[i],
                        mod_features,
                        dropout=dropout,
                        **kwargs,
                    )
                )

                up.append(
                    UNetBlock(
                        hid_channels[i],
                        mod_features,
                        dropout=dropout,
                        **kwargs,
                    )
                )

            # Adding blocks for downsampling and upsampling
            if i > 0:
                do.insert(
                    index=0,
                    module=nn.Sequential(
                        Convolution2DBlock(
                            hid_channels[i - 1],
                            hid_channels[i],
                            stride=stride,
                            **kwargs,
                        ),
                        LayerNorm(dim=1),
                    ),
                )

                up.append(
                    nn.Sequential(
                        LayerNorm(dim=1),
                        UpsampleBlock(scale_factor=float(stride)),
                    )
                )

            # First and last projection blocks
            else:
                do.insert(
                    index=0,
                    module=ConvNd(
                        in_channels,
                        hid_channels[i],
                        3,
                        **kwargs,
                    ),
                )
                up.append(
                    ConvNd(
                        hid_channels[i],
                        out_channels,
                        3,
                        kernel_size=1,  # Removes aliasing artifacts
                    ),
                )

            # Projection for concatening tensors (beginning of each ascent stage)
            if i + 1 < len(hid_blocks):
                up.insert(
                    index=0,
                    module=ConvNd(
                        hid_channels[i] + hid_channels[i + 1],
                        hid_channels[i],
                        3,
                        **kwargs,
                    ),
                )

            # Updating the descent and ascent stages
            self.descent.append(do)
            self.ascent.insert(
                index=0,
                module=up,
            )

    def forward(self, x: Tensor, mod: Tensor) -> Tensor:
        r"""
        Arguments:
            x: Input tensor (B, Ci, T, H, W).
            mod: Modulation vector (B, D).

        Returns:
            Tensor: Output tensor (B, Co, T, H, W).
        """
        # Saves end stage tensors
        memory = []

        # Descent
        for blocks in self.descent:
            for block in blocks:
                if isinstance(block, UNetBlock):
                    x = block(x, mod)
                else:
                    x = block(x)
            memory.append(x)

        # Ascent
        for blocks in self.ascent:
            y = memory.pop()
            if x is not y:
                for i in range(2, x.ndim):
                    if x.shape[i] > y.shape[i]:
                        x = torch.narrow(x, i, 0, y.shape[i])
                x = torch.cat((x, y), dim=1)

            for block in blocks:
                if isinstance(block, UNetBlock):
                    x = block(x, mod)
                else:
                    x = block(x)

        return x
