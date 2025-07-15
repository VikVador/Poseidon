r"""Common layers and modules."""

import torch
import torch.nn as nn

from einops import rearrange
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from typing import Sequence, Union


def ConvNd(
    in_channels: int,
    out_channels: int,
    spatial: int = 2,
    identity_init: bool = False,
    **kwargs,
) -> nn.Module:
    r"""Returns an N-dimensional convolutional layer.

    Arguments:
        in_channels: The number of input channels :math:`C_i`.
        out_channels: The number of output channels :math:`C_o`.
        spatial: The number of spatial dimensions :math:`N`.
        identity_init: Initialize the convolution as a (pseudo-)identity.
        kwargs: Keyword arguments passed to :class:`torch.nn.Conv2d`.
    """

    CONVS = {
        1: nn.Conv1d,
        2: nn.Conv2d,
        3: nn.Conv3d,
    }

    if spatial in CONVS:
        Conv = CONVS[spatial]
    else:
        raise NotImplementedError()

    conv = Conv(in_channels, out_channels, **kwargs)

    if identity_init:
        kernel_size = conv.weight.shape[2:]
        kernel_center = [k // 2 for k in kernel_size]

        eye = torch.zeros_like(conv.weight.data)

        for i in range(out_channels):
            eye[(i, i % in_channels, *kernel_center)] = 1

        conv.weight.data.mul_(1e-2)
        conv.weight.data.add_(eye)

    return conv


class LayerNorm(nn.Module):
    r"""Creates a layer that standardizes features along a dimension.

    .. math:: y = \frac{x - \mathbb{E}[x]}{\sqrt{\mathbb{V}[x] + \epsilon}}

    References:
       | Layer Normalization (Lei Ba et al., 2016)
       | https://arxiv.org/abs/1607.06450

    Arguments:
        dim: The dimension(s) to standardize.
        eps: A numerical stability term.
    """

    def __init__(self, dim: Union[int, Sequence[int]], eps: float = 1e-5):
        super().__init__()

        self.dim = dim if isinstance(dim, int) else tuple(dim)

        self.register_buffer("eps", torch.as_tensor(eps))

    def extra_repr(self) -> str:
        return f"dim={self.dim}"

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The input tensor :math:`x`, with shape :math:(*).

        Returns:
            The standardized tensor :math:`y`, with shape :math:`(*)`.
        """

        variance, mean = torch.var_mean(x, dim=self.dim, keepdim=True)

        return (x - mean) * torch.rsqrt(variance + self.eps)


class SelfAttentionNd(nn.MultiheadAttention):
    r"""Creates an N-dimensional self-attention layer.

    Arguments:
        channels: The number of channels :math:`C`.
        heads: The number of attention heads.
        kwargs: Keyword arguments passed to :class:`torch.nn.MultiheadAttention`.
    """

    def __init__(
        self,
        channels: int,
        heads: int = 1,
        checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__(embed_dim=channels, num_heads=heads, batch_first=True, **kwargs)

        self.checkpointing = checkpointing

    def _forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The input tensor :math:`x`, with shape :math:`(B, C, L_1, ..., L_N)`.

        Returns:
            The ouput tensor :math:`y`, with shape :math:`(B, C, L_1, ..., L_N)`.
        """

        y = rearrange(x, "B C ...  -> B (...) C")

        qkv = torch.nn.functional.linear(y, self.in_proj_weight, self.in_proj_bias)
        q, k, v = rearrange(qkv, "B L (n H C) -> n B H L C", n=3, H=self.num_heads)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout)
        y = rearrange(y, "B H L C -> B L (H C)")
        y = torch.nn.functional.linear(y, self.out_proj.weight, self.out_proj.bias)

        y = rearrange(y, "B L C -> B C L").reshape(x.shape)

        return y

    def forward(self, x: Tensor) -> Tensor:
        if self.checkpointing:
            return checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)
