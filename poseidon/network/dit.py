r"""Diffusion Transformer.

Adapted from:
  | diffusion-transformer (milmor)
  | https://github.com/milmor/diffusion-transformer

"""

import math
import torch
import torch.nn as nn

from torch import Tensor
from typing import (
    Tuple,
)


def modulate(x, shift, scale) -> Tensor:
    """Applies FiLM-like modulation to the input tensor."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PositionalEmbedding(nn.Module):
    """Computes a sinusoidal positional embedding.

    Arguments:
        dim: Dimensionality of the embedding.
        scale: Scaling factor for input values.
    """

    def __init__(
        self,
        dim: int,
        scale: float = 1.0,
    ):
        super().__init__()
        assert dim % 2 == 0, "Embedding dimension must be even."
        self.dim = dim
        self.scale = scale

    def forward(self, x: torch.Tensor) -> Tensor:
        """Computes the sinusoidal positional embedding."""
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LinformerAttention(nn.Module):
    """Implements Linformer self-attention with low-rank projection.

    Arguments:
        seq_len: Sequence length.
        dim: Feature dimension.
        n_heads: Number of attention heads.
        k: Low-rank projection dimension.
        bias: Whether to include bias in linear projections.
    """

    def __init__(
        self,
        seq_len,
        dim,
        n_heads,
        k,
        bias=True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.scale = (dim // n_heads) ** -0.5
        self.qw = nn.Linear(dim, dim, bias=bias)
        self.kw = nn.Linear(dim, dim, bias=bias)
        self.vw = nn.Linear(dim, dim, bias=bias)

        self.E = nn.Parameter(torch.randn(seq_len, k))
        self.F = nn.Parameter(torch.randn(seq_len, k))

        self.ow = nn.Linear(dim, dim, bias=bias)

    def forward(self, x) -> Tensor:
        """Computes Linformer attention."""
        q = self.qw(x)
        k = self.kw(x)
        v = self.vw(x)

        B, L, D = q.shape
        q = torch.reshape(q, [B, L, self.n_heads, -1])
        q = torch.permute(q, [0, 2, 1, 3])
        k = torch.reshape(k, [B, L, self.n_heads, -1])
        k = torch.permute(k, [0, 2, 3, 1])
        v = torch.reshape(v, [B, L, self.n_heads, -1])
        v = torch.permute(v, [0, 2, 3, 1])
        k = torch.matmul(k, self.E[:L, :])

        v = torch.matmul(v, self.F[:L, :])
        v = torch.permute(v, [0, 1, 3, 2])

        qk = torch.matmul(q, k) * self.scale
        attn = torch.softmax(qk, dim=-1)

        v_attn = torch.matmul(attn, v)
        v_attn = torch.permute(v_attn, [0, 2, 1, 3])
        v_attn = torch.reshape(v_attn, [B, L, D])

        x = self.ow(v_attn)

        return x


class TransformerBlock(nn.Module):
    """Transformer block with Linformer attention and FiLM-style conditional modulation.

    Arguments:
        seq_len: Sequence length.
        dim: Feature dimension.
        heads: Number of attention heads.
        mlp_dim: Dimension of MLP hidden layer.
        k: Low-rank projection dimension for Linformer.
        rate: Dropout rate.
    """

    def __init__(
        self,
        seq_len,
        dim,
        heads,
        mlp_dim,
        k,
        rate=0.0,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = LinformerAttention(seq_len, dim, heads, k)
        self.ln_2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_dim),
            nn.GELU(),
            nn.Dropout(rate),
            nn.Linear(dim * mlp_dim, dim),
            nn.Dropout(rate),
        )
        self.gamma_1 = nn.Linear(dim, dim)
        self.beta_1 = nn.Linear(dim, dim)
        self.gamma_2 = nn.Linear(dim, dim)
        self.beta_2 = nn.Linear(dim, dim)
        self.scale_1 = nn.Linear(dim, dim)
        self.scale_2 = nn.Linear(dim, dim)

        # Initialize weights to zero for modulation and gating layers
        self._init_weights([
            self.gamma_1,
            self.beta_1,
            self.gamma_2,
            self.beta_2,
            self.scale_1,
            self.scale_2,
        ])

    def _init_weights(self, layers):
        for layer in layers:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, c) -> Tensor:
        """Applies the Transformer block with conditioning vector c."""
        scale_msa = self.gamma_1(c)
        shift_msa = self.beta_1(c)
        scale_mlp = self.gamma_2(c)
        shift_mlp = self.beta_2(c)
        gate_msa = self.scale_1(c).unsqueeze(1)
        gate_mlp = self.scale_2(c).unsqueeze(1)
        x = self.attn(modulate(self.ln_1(x), shift_msa, scale_msa)) * gate_msa + x
        return self.mlp(modulate(self.ln_2(x), shift_mlp, scale_mlp)) * gate_mlp + x


class DiT(nn.Module):
    """Diffusion Transformer model."

    Arguments:
        depth: Number of transformer blocks.
        heads: Number of attention heads.
        mlp_ratio: Ratio of MLP hidden dimension to input dimension.
        embedding: Dimensionality of the embedding (diffusion step).
        embedding_lfa: Projection dimension of linformer attention.
        dimensions: Input tensor dimensions (B, C, K, X, Y).
    """

    def __init__(
        self,
        depth: int,
        heads: int,
        mlp_ratio: int,
        embedding: int,
        embedding_lfa: int,
        dimensions: Tuple[int, int, int, int, int],
    ):
        super(DiT, self).__init__()

        self.B, self.C, self.K, self.X, self.Y = dimensions

        # Initializations
        self.depth, self.token_nb, self.token_dim = (
            depth,
            self.X * self.Y,
            self.C * self.K,
        )

        # Positional embedding of tokens (B, X * Y, C * K)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.token_nb, self.token_dim))

        # Transformer blocks
        self.transformer = nn.ModuleList([
            TransformerBlock(self.token_nb, self.token_dim, heads, mlp_ratio, embedding_lfa)
            for _ in range(depth)
        ])

        # Diffusion timestep modulation
        self.time_emb = nn.Sequential(
            PositionalEmbedding(embedding),
            nn.Linear(embedding, embedding),
            nn.SiLU(),
            nn.Linear(embedding, self.token_dim),
        )

    def forward(self, x, t) -> Tensor:
        """Forward pass of the model."""

        # Projecting diffusion timestep
        t = self.time_emb(t)

        # Adding positional embedding
        x += self.pos_embedding

        # Applying transformer blocks
        for layer in self.transformer:
            x = layer(x, t)

        return x
