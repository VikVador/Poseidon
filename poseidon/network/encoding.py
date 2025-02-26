r"""Encoding blocks."""

import torch
import torch.nn as nn

from torch import Tensor


class SineEncoding(nn.Module):
    r"""Creates a sinusoidal positional encoding.

    References:
        | Attention Is All You Need (Vaswani et al., 2017)
        | https://arxiv.org/abs/1706.03762

    Arguments:
        features: Number of embedding features (F). Must be even.
        omega: Maximum frequency omega.
    """

    def __init__(
        self,
        features: int,
        omega: float = 1e3,
    ):
        super().__init__()

        assert features % 2 == 0, "ERROR (SineEncoding) - The number of features must be even."
        freqs = torch.linspace(0, 1, features // 2, dtype=torch.float64)
        freqs = omega ** (-freqs)
        self.freqs = freqs.to(dtype=torch.float32)

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: Tensor to encode (*)

        Returns:
            Embedded tensor (*, F)
        """
        x = x[..., None]
        return torch.cat(
            (
                torch.sin(self.freqs * x),
                torch.cos(self.freqs * x),
            ),
            dim=-1,
        )
