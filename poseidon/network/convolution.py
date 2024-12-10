r"""Convolutional blocks."""

import torch.nn as nn

from torch import Tensor

# isort: split
from poseidon.network.tools import reshape, unshape


def ConvNd(in_channels: int, out_channels: int, spatial: int, **kwargs) -> nn.Module:
    r"""Returns an N-dimensional convolutional layer.

    Arguments:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        spatial: Number of spatial dimensions on which the convolution is applied.
        kwargs: Keyword arguments passed to :class:`torch.nn.Conv2d`.
    """
    if spatial not in {1, 2, 3}:
        raise NotImplementedError(
            f"Unsupported spatial dimension {spatial}. Supported dimensions: 1, 2, 3."
        )
    Conv = getattr(nn, f"Conv{spatial}d")
    return Conv(in_channels, out_channels, **kwargs)


class Convolution2DBlock(nn.Module):
    r"""A spatial convolutional block for a 5-dimensional input.

    Information
        This block reshapes a 5-dimensional input (B, C, T, X, Y) into a format
        suitable for 2-dimensional convolutions (B * T, C, X, Y) along the
        spatial dimensions.

    Arguments:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        **kwargs: Keyword arguments passed to :class:`torch.nn.Conv2d`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs,
    ):
        super().__init__()
        self.block = ConvNd(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial=2,
            **kwargs,
        )

    def forward(self, x: Tensor) -> Tensor:
        x, _, original_shape = reshape(hide="time", x=x)
        x = self.block(x)
        x = unshape(
            extract="time",
            x=x,
            shape=original_shape[:3] + x.shape[-2:],
        )
        return x
