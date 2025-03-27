r"""Diffusion Transformer.

Adapted from:
  | diffusion-transformer (milmor)
  | https://github.com/milmor/diffusion-transformer

"""

import math
import torch
import torch.nn as nn

from torch import Tensor


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
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(rate),
            nn.Linear(mlp_dim, dim),
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


class FinalLayer(nn.Module):
    """Final layer of the model with linear projection and modulation.

    Arguments:
        dim: Feature dimension.
        patch_size: Size of the patches.
        out_channels: Number of output channels.
    """

    def __init__(
        self,
        dim,
        patch_size,
        out_channels,
    ):
        super().__init__()
        self.ln_final = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels, bias=True)
        self.gamma = nn.Linear(dim, dim)
        self.beta = nn.Linear(dim, dim)

        # Initialize weights and biases to zero for modulation and linear layers
        self._init_weights([self.linear, self.gamma, self.beta])

    def _init_weights(self, layers):
        for layer in layers:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, c) -> Tensor:
        """Applies the final layer with conditioning vector c."""
        scale = self.gamma(c)
        shift = self.beta(c)
        x = modulate(self.ln_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """Diffusion Transformer model."

    Arguments:
        img_size: Size of the input image.
        dim: Feature dimension.
        patch_size: Size of the patches.
        depth: Number of transformer blocks.
        heads: Number of attention heads.
        mlp_dim: Dimension of MLP hidden layer.
        k: Low-rank projection dimension for Linformer.
        in_channels: Number of input channels.
    """

    def __init__(
        self,
        img_size,
        dim=64,
        patch_size=4,
        depth=3,
        heads=4,
        mlp_dim=512,
        k=64,
        in_channels=3,
    ):
        super(DiT, self).__init__()
        self.dim = dim
        self.n_patches = (img_size // patch_size) ** 2
        self.depth = depth
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_patches, dim))
        self.patches = nn.Sequential(
            nn.Conv2d(
                in_channels, dim, kernel_size=patch_size, stride=patch_size, padding=0, bias=False
            ),
        )

        self.transformer = nn.ModuleList()
        for _ in range(self.depth):
            self.transformer.append(TransformerBlock(self.n_patches, dim, heads, mlp_dim, k))

        self.emb = nn.Sequential(
            PositionalEmbedding(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        self.final = FinalLayer(dim, patch_size, in_channels)
        self.ps = nn.PixelShuffle(patch_size)

    def forward(self, x, t) -> Tensor:
        """Forward pass of the model."""
        t = self.emb(t)
        x = self.patches(x)
        B, C, H, W = x.shape
        x = x.permute([0, 2, 3, 1]).reshape([B, H * W, C])
        x += self.pos_embedding
        for layer in self.transformer:
            x = layer(x, t)

        x = self.final(x, t).permute([0, 2, 1])
        x = x.reshape([B, -1, H, W])
        x = self.ps(x)
        return x
