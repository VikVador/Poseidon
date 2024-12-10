import torch
import torch.nn as nn

from torch import Tensor
from typing import Optional, Sequence, Union

# isort: split
from poseidon.network.convolution import Convolution2DBlock
from poseidon.network.normalization import LayerNorm
from poseidon.network.residuals import (
    SpatialModulatedResidualBlock,
    TemporalModulatedResidualBlock,
)
from poseidon.network.upsample import UpsampleBlock


class UNetBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        mod_features: int,
        dropout: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()

        # Spatial and temporal processing blocks
        self.block_spatial = SpatialModulatedResidualBlock(
            channels=channels,
            mod_features=mod_features,
            dropout=dropout,
            **kwargs,
        )
        self.block_temporal = TemporalModulatedResidualBlock(
            channels=channels,
            mod_features=mod_features,
            dropout=dropout,
            **kwargs,
        )

    def forward(self, x: Tensor, mod: Tensor) -> Tensor:
        x = self.block_spatial(x, mod)
        x = self.block_temporal(x, mod)
        return x


class UNet(nn.Module):
    r"""Creates a modulated U-Net module.

    Arguments:
        in_channels: The number of input channels :math:`C_i`.
        out_channels: The number of output channels :math:`C_o`.
        mod_features: The number of modulating features :math:`D`.
        hid_channels: The numbers of channels at each depth.
        hid_blocks: The numbers of hidden blocks at each depth.
        kernel_size: The kernel size of all convolutions.
        stride: The stride of the downsampling convolutions.
        attention_heads: The number of attention heads at each depth.
        dropout: The dropout rate in :math:`[0, 1]`.
        spatial: The number of spatial dimensions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mod_features: int,
        hid_channels: Sequence[int] = (64, 128, 256),
        hid_blocks: Sequence[int] = (3, 3, 3),
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()

        assert len(hid_blocks) == len(hid_channels)

        kwargs = dict(
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.descent, self.ascent = nn.ModuleList(), nn.ModuleList()

        for i, blocks in enumerate(hid_blocks):
            do, up = nn.ModuleList(), nn.ModuleList()

            # BLOCKS INSIDE A STAGE OF THE UNET
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

            # ADDING BLOCKS FOR DOWNSAMPLING AND UPSAMPLING
            if i > 0:
                do.insert(
                    0,
                    nn.Sequential(
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
                        UpsampleBlock(scale_factor=tuple(stride)),
                    )
                )

            # ADDING FIST EVER BLOCK FOR PROJECTION ON CORRECT STRATING CHANNELS AND FINAL BLOCK FOR OUTPUT
            else:
                do.insert(0, Convolution2DBlock(in_channels, hid_channels[i], **kwargs))
                up.append(Convolution2DBlock(hid_channels[i], out_channels, kernel_size=1))

            # CONVOLUTION ON THE CONCATENATED CHANNELS TO GET BACK TO CHANNEL PER STAGE
            if i + 1 < len(hid_blocks):
                up.insert(
                    0,
                    Convolution2DBlock(
                        hid_channels[i] + hid_channels[i + 1],
                        hid_channels[i],
                        **kwargs,
                    ),
                )

            self.descent.append(do)
            self.ascent.insert(0, up)

    def forward(self, x: Tensor, mod: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The input tensor, with shape :math:`(B, C_i, H_1, ..., H_N)`.
            mod: The modulation vector, with shape :math:`(D)` or :math:`(B, D)`.

        Returns:
            The output tensor, with shape :math:`(B, C_o, H_1, ..., H_N)`.
        """

        memory = []

        for blocks in self.descent:
            for block in blocks:
                if isinstance(block, UNetBlock):
                    x = block(x, mod)
                else:
                    x = block(x)

            memory.append(x)

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
