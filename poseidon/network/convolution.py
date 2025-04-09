r"""Convolutional blocks."""

import torch
import torch.nn as nn

from torch import Tensor
from typing import Dict, Optional

# isort: split
from poseidon.network.attention import SelfAttentionNd
from poseidon.network.embedding import MeshEmbedding
from poseidon.network.encoding import SineEncoding
from poseidon.network.modulation import Modulator
from poseidon.network.normalization import LayerNorm


def ConvNd(
    in_channels: int,
    out_channels: int,
    spatial: int,
    **kwargs,
) -> nn.Module:
    r"""Returns an N-dimensional convolutional layer.

    Arguments:
        in_channels: Number of input channels (C_i).
        out_channels: Number of output channels (C_o).
        spatial: Number of spatial dimensions on which the convolution is applied.
        kwargs: Keyword arguments passed to :class:`torch.nn.Conv2d`.
    """
    if spatial not in {1, 2, 3}:
        raise NotImplementedError(
            f"Unsupported spatial dimension {spatial}. Supported dimensions: 1, 2, 3."
        )
    Conv = getattr(nn, f"Conv{spatial}d")
    return Conv(in_channels, out_channels, **kwargs)


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
            ConvNd(
                channels,
                channels * ffn_scaling,
                spatial=3,
                **kwargs,
            ),
            nn.SiLU(),
            ConvNd(
                channels * ffn_scaling,
                channels,
                spatial=3,
                **kwargs,
            ),
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
            LayerNorm(dim=1),
            SineEncoding(features=mod_features),
            Modulator(
                channels=channels,
                mod_features=mod_features,
                spatial=3,
            ),
        )

    def forward(
        self,
        x: Tensor,
        mod: Tensor,
    ) -> Tensor:
        r"""
        Arguments:
            x: Input tensor (B, C_i, K, X, Y).
            mod: Modulation vector (B, D).

        Returns:
            Tensor: Output tensor (B, C_o, K, X, Y).
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
