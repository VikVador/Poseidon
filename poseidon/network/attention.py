r"""Attention blocks."""

import torch.nn as nn

from einops import rearrange
from torch import Tensor


class SelfAttentionNd(nn.MultiheadAttention):
    r"""Creates an N-dimensional self-attention layer.

    Arguments:
        channels: Number of channels (C)
        heads: Number of attention heads (N).
        kwargs: Keyword arguments passed to `torch.nn.MultiheadAttention`.
    """

    def __init__(self, channels: int, heads: int = 1, **kwargs):
        super().__init__(embed_dim=channels, num_heads=heads, batch_first=True, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: Input tensor shape (B, C, T, H, W).

        Returns:
            Ouput tensor (B, C, T, H, W).
        """

        y = rearrange(x, "B C ...  -> B (...) C")
        y, _ = super().forward(y, y, y, average_attn_weights=False)
        y = rearrange(y, "B L C -> B C L").reshape(x.shape)

        return y
