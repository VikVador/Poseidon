r"""UNet Architecture."""

import torch
import torch.nn as nn

from torch import Tensor
from typing import Dict, Optional, Sequence

# isort: split
from poseidon.network.attention import SelfAttentionNd
from poseidon.network.convolution import ConvNd
from poseidon.network.embedding import MeshEmbedding
from poseidon.network.encoding import SineEncoding
from poseidon.network.modulation import Modulator
from poseidon.network.normalization import LayerNorm


class UNetBlock(nn.Module):
    r"""Creates a UNet residual block.

    Arguments:
        channels: Number of channels (C) in the input tensor.
        mod_features: Number of features (D) in the modulation vector.
        ffn_scaling: Scaling factor for the feed-forward network.
        spatial_scaling: Scaling factor for the spatial region.
        config_region: Configuration for the spatial region.
        config_siren: Configuration for the Siren architecture.
        attention_heads: Number of attention heads.
        dropout: Dropout probability for regularization [0, 1].
        **kwargs: Additional arguments passed to the residual blocks.
    """

    def __init__(
        self,
        channels: int,
        mod_features: int,
        ffn_scaling: int,
        spatial_scaling: int,
        config_region: Dict,
        config_siren: Dict,
        attention_heads: Optional[int] = None,
        dropout: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()

        # Attention
        self.attn = (
            SelfAttentionNd(channels, heads=attention_heads)
            if attention_heads is not None
            else None
        )

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            ConvNd(channels, channels * ffn_scaling, spatial=2, **kwargs),
            nn.SiLU(),
            nn.Identity() if dropout is None else nn.Dropout(dropout),
            ConvNd(channels * ffn_scaling, channels, spatial=2, **kwargs),
        )

        # Spatial mesh embedding
        self.mesh_embedding = MeshEmbedding(
            channels=channels,
            spatial_scaling=spatial_scaling,
            config_region=config_region,
            **config_siren,
        )

        # Modulator
        self.norm, self.encoder, self.modulator = (
            LayerNorm(1),
            SineEncoding(features=mod_features),
            Modulator(
                channels=channels,
                mod_features=mod_features,
                spatial=2,
            ),
        )

    def forward(
        self,
        x: Tensor,
        mod: Tensor,
    ) -> Tensor:
        r"""
        Arguments:
            x: Input tensor (B, (C * K), X, Y).
            mod: Modulation vector (B, D).

        Returns:
            Tensor: Output tensor (B, (C * K), X, Y).
        """

        # Encoding modulation vector
        mod = self.encoder(mod).squeeze(1)

        # Mesh embedding
        mesh = self.mesh_embedding()

        # Modulation
        a, b, c = self.modulator(mod)

        y = (a + 1) * self.norm(x + mesh) + b
        y = y if self.attn is None else y + self.attn(y)
        y = self.ffn(y)
        y = (x + c * y) * torch.rsqrt(1 + c * c)

        return y


class UNet(nn.Module):
    r"""Creates a U-Net.

    Arguments:
        in_channels: Number of input channels (C_i)
        out_channels: Number of output channels (C_o).
        kernel_size: Kernel size of all convolutions.
        mod_features: Number of features (D) in modulating vector.
        ffn_scaling: Scaling factor for the feed-forward network.
        config_region: Configuration for the spatial region.
        config_siren: Configuration for the Siren architecture.
        stride: Stride of the spatial downsampling convolutions.
        dropout: Dropout probability for regularization [0, 1].
        hid_channels: Numbers of channels at each depth.
        hid_blocks: Numbers of hidden blocks at each depth.
        attention_heads: The number of attention heads at each depth.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        mod_features: int,
        ffn_scaling: int,
        config_region: Dict,
        config_siren: Dict,
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
                kernel_size,
                kernel_size,
            ),
            padding=(
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
                        ffn_scaling,
                        i,
                        config_region,
                        config_siren,
                        attention_heads.get(str(i), None),
                        dropout,
                        **kwargs,
                    )
                )
                up.append(
                    UNetBlock(
                        hid_channels[i],
                        mod_features,
                        ffn_scaling,
                        i,
                        config_region,
                        config_siren,
                        attention_heads.get(str(i), None),
                        dropout,
                        **kwargs,
                    )
                )

            # Downsampling and Upsampling
            if i > 0:
                do.insert(
                    index=0,
                    module=nn.Sequential(
                        ConvNd(
                            hid_channels[i - 1],
                            hid_channels[i],
                            spatial=2,
                            stride=(stride, stride),
                            **kwargs,
                        ),
                    ),
                )

                up.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=(stride, stride), mode="nearest"),
                    )
                )

            # Projections - Initial and final stages
            else:
                do.insert(
                    0,
                    ConvNd(
                        in_channels,
                        hid_channels[i],
                        spatial=2,
                        **kwargs,
                    ),
                )
                up.append(
                    ConvNd(
                        hid_channels[i],
                        out_channels,
                        spatial=2,
                        **kwargs,
                    )
                )

            # Projection - Connecting stages
            if i + 1 < len(hid_blocks):
                up.insert(
                    0,
                    ConvNd(
                        hid_channels[i] + hid_channels[i + 1],
                        hid_channels[i],
                        spatial=2,
                        **kwargs,
                    ),
                )

            # Updating the descent and ascent stages
            self.descent.append(do)
            self.ascent.insert(0, up)

    def forward(
        self,
        x: Tensor,
        mod: Tensor,
    ) -> Tensor:
        r"""A forward pass through the UNet.

        Arguments:
            x: Input tensor (B, C_in, X, Y).
            mod: Modulation vector (B, D).

        Returns:
            Output tensor (B, C_out, X, Y).
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
