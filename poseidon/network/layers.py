r"""Network - Shared components for Neural Network module"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    r"""Creates a normalization layer.

    Arguments:
        dim: The dimension(s) along which the normalization is performed.
        eps: A constant that prevents numerical instabilities.
    """

    def __init__(self, dim: int = -1, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Performs a forward pass through the normalization layer.

        Arguments:
            x: The input tensor (*).

        Returns:
            The input tensor normalized along specified dimension(s) (*).
        """
        v, m = torch.var_mean(x, dim=self.dim, keepdim=True)
        return (x - m) / torch.sqrt(v + self.eps)


class PosEmbedding(nn.Module):
    r"""Creates a positional embedding module.

    References:
        | Attention Is All You Need (Vaswani et al., 2017).
        | https://arxiv.org/abs/1706.03762

    Arguments:
        features: The number of embedding features (F).
    """

    def __init__(self, features: int):
        super().__init__()
        freqs = torch.linspace(0, 1, features // 2)
        freqs = (1 / 1e4) ** freqs
        self.register_buffer("freqs", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Arguments:
            x: The position tensor (*).

        Returns:
            The position embedding (*, F).
        """
        x = x[..., None]
        return torch.cat(
            (
                torch.sin(self.freqs * x),
                torch.cos(self.freqs * x),
            ),
            dim=-1,
        )
