r"""Convolutional blocks."""

import torch.nn as nn


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
