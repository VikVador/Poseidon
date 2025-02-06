r"""UNet Architecture for 3-dimensional convolutions.

References:
    Inspired by the implementation in:
    https://github.com/probabilists/azula
"""

import torch
import torch.nn as nn

from torch import Tensor
from typing import Dict, Optional, Sequence

# isort: split
from poseidon.network.attention import SelfAttentionNd
from poseidon.network.convolution import ConvNd
from poseidon.network.modulation import Modulator
from poseidon.network.normalization import LayerNorm


class UNetBlock(nn.Module):
    r"""Creates a UNet residual block.

    Arguments:
        channels: Number of channels (C) in the input tensor.
        mod_features: Number of features (D) in the modulation vector.
        dropout: Dropout probability for regularization [0, 1].
        attention_heads: Number of attention heads.
        **kwargs: Additional arguments passed to the residual blocks.
    """

    def __init__(
        self,
        channels: int,
        mod_features: int,
        attention_heads: Optional[int] = None,
        dropout: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()

        self.modulator = Modulator(
            channels=channels,
            mod_features=mod_features,
            spatial=3,
        )

        self.block = nn.Sequential(
            LayerNorm(dim=1),
            ConvNd(
                channels,
                channels,
                spatial=3,
                **kwargs,
            ),
            nn.SiLU(),
            nn.Identity() if dropout is None else nn.Dropout(dropout),
            ConvNd(
                channels,
                channels,
                spatial=3,
                **kwargs,
            ),
        )

        if attention_heads is not None:
            self.block.append(LayerNorm(dim=1))
            self.block.append(SelfAttentionNd(channels, heads=attention_heads))

    def forward(self, x: Tensor, mod: Tensor) -> Tensor:
        r"""
        Arguments:
            x: Input tensor (B, C, T, H, W).
            mod: Modulation vector (B, D).

        Returns:
            Tensor: Output tensor (B, C, T, H, W).
        """
        mod_factor, mod_bias, mod_scaling = self.modulator(mod)

        y = (mod_factor + 1) * x + mod_bias
        y = self.block(y)
        y = x + mod_scaling * y
        y = y / torch.sqrt(1 + mod_scaling * mod_scaling)

        return y


class UNet(nn.Module):
    r"""Creates a U-Net model for 3-dimensional convolutions.

    Arguments:
        in_channels: Number of input channels (C_i)
        out_channels: Number of output channels (C_o).
        mod_features: Number of features (D) in modulating vector.
        kernel_size: Kernel size of all convolutions.
        blanket_size: Size of the temporal convolution.
        stride: Stride of the spatial downsampling convolutions.
        dropout: Dropout probability for regularization [0, 1].
        hid_channels: Numbers of channels at each depth.
        hid_blocks: Numbers of hidden blocks at each depth.
        attention_heads: The number of attention heads at each depth.

    Example:
        >>> unet = UNet(in_channels=64,
                        out_channels=64,
                        mod_features=128,
                        hid_channels=[64, 128, 256],
                        hid_blocks=[2, 3, 3],
                        kernel_size=3,
                        blanket_size=3,
                        stride=2,
                        dropout=0.1)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mod_features: int,
        kernel_size: int,
        blanket_size: int,
        stride: int = 2,
        dropout: Optional[float] = None,
        hid_blocks: Sequence[int] = (1, 1, 1),
        hid_channels: Sequence[int] = (32, 64, 128),
        attention_heads: Dict[str, int] = {},  # noqa: B006
    ):
        super().__init__()

        assert len(hid_blocks) == len(
            hid_channels
        ), "ERROR (UNet) - Mismatched number of hidden blocks and channels."

        kwargs = dict(
            kernel_size=(
                blanket_size,
                kernel_size,
                kernel_size,
            ),
            padding=(
                blanket_size // 2,
                kernel_size // 2,
                kernel_size // 2,
            ),
        )

        # Contains the blocks for the descent and ascent of the UNet
        self.descent, self.ascent = nn.ModuleList(), nn.ModuleList()

        for i, blocks in enumerate(hid_blocks):
            # Contains the blocks for the descent and ascent of the UNet
            do, up = nn.ModuleList(), nn.ModuleList()

            # Blocks - Descent and Ascent Residual Blocks
            for _ in range(blocks):
                do.append(
                    UNetBlock(
                        hid_channels[i],
                        mod_features,
                        attention_heads.get(str(i), None),
                        dropout,
                        **kwargs,
                    )
                )
                up.append(
                    UNetBlock(
                        hid_channels[i],
                        mod_features,
                        attention_heads.get(str(i), None),
                        dropout,
                        **kwargs,
                    )
                )

            # Sampling - Downsampling and Upsampling
            if i > 0:
                do.insert(
                    index=0,
                    module=nn.Sequential(
                        ConvNd(
                            hid_channels[i - 1],
                            hid_channels[i],
                            spatial=3,
                            stride=(1, stride, stride),
                            **kwargs,
                        ),
                        LayerNorm(dim=1),
                    ),
                )

                up.append(
                    nn.Sequential(
                        LayerNorm(dim=1),
                        nn.Upsample(scale_factor=(1, stride, stride), mode="nearest"),
                    )
                )

            # Projections - Initial and final stages
            else:
                do.insert(
                    0,
                    ConvNd(
                        in_channels,
                        hid_channels[i],
                        spatial=3,
                        **kwargs,
                    ),
                )
                up.append(
                    ConvNd(
                        hid_channels[i],
                        out_channels,
                        spatial=3,
                        kernel_size=(blanket_size, 1, 1),  # Removes aliasing
                        padding=(blanket_size // 2, 0, 0),
                    )
                )

            # Projection - Connecting stages
            if i + 1 < len(hid_blocks):
                up.insert(
                    0,
                    ConvNd(
                        hid_channels[i] + hid_channels[i + 1],
                        hid_channels[i],
                        spatial=3,
                        **kwargs,
                    ),
                )

            # Updating the descent and ascent stages
            self.descent.append(do)
            self.ascent.insert(0, up)

    def forward(self, x: Tensor, mod: Tensor) -> Tensor:
        r"""A forward pass through the UNet.

        Arguments:
            x: Input tensor (B, Ci, T, H, W).
            mod: Modulation vector (B, D).

        Returns:
            Tensor: Output tensor (B, Co, T, H, W).
        """

        memory = []

        # === Descent ===
        for blocks in self.descent:
            for block in blocks:
                if isinstance(block, UNetBlock):
                    x = block(x, mod)
                else:
                    x = block(x)

            memory.append(x)

        # === Ascent ===
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
