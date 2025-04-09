r"""Attention blocks."""

import torch.nn as nn

from einops import rearrange
from torch import Tensor


class SelfAttentionNd(nn.MultiheadAttention):
    r"""Creates an N-dimensional self-attention layer.

    Arguments:
        channels: Number of channels (C)
        heads: Number of attention heads (N).
        channel_first: If True, the input tensor shape expected (B, C, ...) otherwise (B, ..., C).
        kwargs: Keyword arguments passed to :class:`torch.nn.MultiheadAttention`.
    """

    def __init__(
        self,
        channels: int,
        heads: int = 1,
        channel_first: bool = True,
        **kwargs,
    ):
        super().__init__(embed_dim=channels, num_heads=heads, batch_first=True, **kwargs)
        self.channel_first = channel_first

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: Input (B, C, ...) or (B, ..., C) if `channel_first` is False.

        Returns:
            Tensor (B, C, ...).
        """

        # Channels must be the last dimension
        y = rearrange(x, "B C ...  -> B (...) C") if self.channel_first else x

        # Multihead self-attention
        y, _ = super().forward(y, y, y, average_attn_weights=False)

        # Rearrange back to original shape
        return rearrange(y, "B L C -> B C L").reshape(x.shape) if self.channel_first else y
