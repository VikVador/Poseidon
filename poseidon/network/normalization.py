r"""Normalization blocks."""

import torch
import torch.nn as nn

from torch import Tensor
from typing import Sequence, Union


class LayerNorm(nn.Module):
    r"""Creates a layer that standardizes features along a dimension.

    References:
       | Layer Normalization (Lei Ba et al., 2016)
       | https://arxiv.org/abs/1607.06450

    Arguments:
        dim: Dimension(s) along which to standardize.
        eps: Small value added for numerical stability.
    """

    def __init__(
        self,
        dim: Union[int, Sequence[int]],
        eps: float = 1e-5,
    ):
        super().__init__()

        self.dim = dim if isinstance(dim, int) else tuple(dim)
        self.register_buffer("eps", torch.as_tensor(eps))

    def forward(self, x: Tensor) -> Tensor:
        r"""Standardizes the input tensor along the specified dimension(s).

        Arguments:
            x: Input tensor.

        Returns:
            Standardized tensor.
        """
        variance, mean = torch.var_mean(x, dim=self.dim, keepdim=True)
        return (x - mean) / (variance + self.eps).sqrt()
