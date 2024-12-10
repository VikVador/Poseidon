r"""Upsampling blocks."""

import torch.nn as nn

from torch import Tensor

# isort: split
from poseidon.network.tools import reshape, unshape


class UpsampleBlock(nn.Module):
    r"""Upsample a 5-dimensional tensor along spatial dimensions.

    Arguments:
        scale_factor: Factor (S) by which to upsample the spatial dimensions.
        mode: Interpolation mode for upsampling.
        kwargs: Additional keyword arguments for `nn.Upsample`.
    """

    def __init__(self, scale_factor: float, mode: str = "nearest", **kwargs):
        super().__init__()
        self.upsampling_block = nn.Upsample(
            scale_factor=scale_factor,
            mode=mode,
            **kwargs,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Input tensor of shape (B, C, T, H, W).

        Returns:
            Tensor: Upsampled input of shape (B, C, T, H * S, W * S).
        """
        x, _, original_shape = reshape(hide="time", x=x)
        x = self.upsampling_block(x)
        return unshape(
            extract="time",
            x=x,
            shape=original_shape[:3] + x.shape[-2:],  # Updating with upsampled spatial dimensions
        )
