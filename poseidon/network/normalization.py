r"""Normalization blocks.

Inspired by: https://github.com/probabilists/azula

"""

import torch
import torch.nn as nn

from torch import Tensor
from typing import Sequence, Union


class LayerNorm(nn.Module):
    r"""Creates a layer that standardizes features along a dimension.

    The output is computed by subtracting the mean and dividing by the square root of the variance
    (with an added small value for numerical stability).

    References:
       | Layer Normalization (Lei Ba et al., 2016)
       | https://arxiv.org/abs/1607.06450

    Arguments:
        dim: The dimension(s) along which to standardize.
        eps: A small value added for numerical stability.
    """

    def __init__(self, dim: Union[int, Sequence[int]], eps: float = 1e-5):
        super().__init__()

        self.dim = dim if isinstance(dim, int) else tuple(dim)

        self.register_buffer("eps", torch.as_tensor(eps))

    def extra_repr(self) -> str:
        return f"dim={self.dim}"

    def forward(self, x: Tensor) -> Tensor:
        r"""Standardizes the input tensor along the specified dimension(s).

        Arguments:
            x: The input tensor with any shape.

        Returns:
            A standardized tensor with the same shape as the input.
        """
        variance, mean = torch.var_mean(x, dim=self.dim, keepdim=True)
        return (x - mean) / (variance + self.eps).sqrt()
