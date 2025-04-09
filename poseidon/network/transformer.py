r"""Transformer blocks."""

import torch
import torch.nn as nn

from einops import rearrange
from torch import Tensor

# isort: split
from poseidon.network.attention import SelfAttentionNd
from poseidon.network.encoding import SineEncoding
from poseidon.network.modulation import Modulator


class Patchify(nn.Module):
    r"""Transforms spatial dimensions into patches."""

    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x: Tensor) -> Tensor:
        r"""Generates patches from spatial dimensions."""
        return rearrange(
            x, "B C K (X x) (Y y) -> B (C x y) K X Y", x=self.patch_size, y=self.patch_size
        )


class Unpatchify(nn.Module):
    r"""Transforms patches back to spatial dimensions."""

    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x: Tensor) -> Tensor:
        r"""Transforms patches back to spatial dimensions."""
        return rearrange(
            x, "B (C x y) K X Y -> B C K (X x) (Y y)", x=self.patch_size, y=self.patch_size
        )


class TransformerBlock(nn.Module):
    r"""Creates a transformer block."""

    def __init__(
        self,
        channels: int,
        mod_features: int,
        ffn_scaling: int,
        heads: int,
    ):
        super().__init__()

        # Attention
        self.attn = SelfAttentionNd(
            channels=channels,
            heads=heads,
            channel_first=False,
        )

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * ffn_scaling),
            nn.SiLU(),
            nn.Linear(channels * ffn_scaling, channels),
        )

        # Modulator
        self.norm, self.modulator = (
            nn.LayerNorm(channels, elementwise_affine=False),
            Modulator(
                channels=channels,
                mod_features=mod_features,
                spatial=1,
            ),
        )

    def forward(
        self,
        x: Tensor,
        mod: Tensor,
    ) -> Tensor:
        r"""
        Arguments:
            x: Input tensor (B, N, C).
            mod: Modulation vector (B, D).

        Returns:
            Tensor (B, N, C).
        """

        # Modulation
        a, b, c = self.modulator(mod)
        a, b, c = (
            rearrange(a, "B C 1 -> B 1 C"),
            rearrange(b, "B C 1 -> B 1 C"),
            rearrange(c, "B C 1 -> B 1 C"),
        )

        y = (a + 1) * self.norm(x) + b
        y = y + self.attn(y)
        y = self.ffn(y)
        y = (x + c * y) * torch.rsqrt(1 + c * c)

        return y


class Transformer(nn.Module):
    r"""Creates a transformer.

    Arguments:
        in_channels: Number of input channels (C_i).
        mod_features: Number of features (D) in modulating vector.
        ffn_scaling: Scaling factor for the feed-forward network.
        hid_channels: Numbers of channels at each depth.
        hid_blocks: Numbers of hidden blocks at each depth.
        patch_size: Size of patches to be generated.
        heads: Number of attention heads at each depth.
    """

    def __init__(
        self,
        in_channels: int,
        mod_features: int,
        ffn_scaling: int,
        hid_channels: int,
        hid_blocks: int,
        patch_size: int,
        attention_heads: int,
    ):
        super().__init__()

        # Encodes diffusion timestep
        self.encoder = SineEncoding(
            features=mod_features,
        )

        self.patchify, self.unpatchify = (
            Patchify(patch_size=patch_size),
            Unpatchify(patch_size=patch_size),
        )

        self.project, self.unproject = (
            nn.Linear(
                in_features=in_channels,
                out_features=hid_channels,
            ),
            nn.Linear(
                in_features=hid_channels,
                out_features=in_channels,
            ),
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                channels=hid_channels,
                mod_features=mod_features,
                ffn_scaling=ffn_scaling,
                heads=attention_heads,
            )
            for _ in range(hid_blocks)
        ])

    def forward(self, x: Tensor, mod: Tensor) -> Tensor:
        r"""Forward pass through the transformer.

        Arguments:
            x: Input tensor (B, C, K, X, Y).
            mod: Modulation vector (B, D).

        Returns:
            Tensor: Output tensor (B, C, K, X, Y).
        """

        # Encoding modulation vector
        mod = self.encoder(mod).squeeze(1)

        # Transforming spatial dimensions into patches
        x = self.patchify(x)

        # Extracting dimensions
        _, C, K, X, Y = x.shape

        # Projecting to transformer channels
        x = self.project(
            rearrange(x, "B C K X Y -> B (X Y) (C K)"),
        )

        # Going through transformer blocks
        for block in self.blocks:
            x = block(x, mod)

        # Projecting back to original UNet channels
        x = rearrange(
            self.unproject(x),
            "B (X Y) (C K) -> B C K X Y",
            X=X,
            Y=Y,
            C=C,
            K=K,
        )

        # Reshape back to original dimensions
        return self.unpatchify(x)
