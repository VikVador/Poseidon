r"""Neural networks"""

import torch
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor
from typing import Dict, Sequence, Tuple


def ConvNd(in_channels: int, out_channels: int, kernel_size: Sequence[int], **kwargs) -> nn.Module:
    r"""Creates a N-dimensional convolutional layer.

    The number of spatial dimensions is inferred from kernel size.

    Arguments:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: The size of the kernel along each dimension.
        kwargs: Keyword arguments passed to :class:`nn.ConvNd`.
    """

    if len(kernel_size) == 1:
        return nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs)
    elif len(kernel_size) == 2:
        return nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
    elif len(kernel_size) == 3:
        return nn.Conv3d(in_channels, out_channels, kernel_size, **kwargs)
    else:
        raise NotImplementedError()


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

    def forward(self, x: Tensor):
        r"""
        Arguments:
            x: The input tensor, with shape :math:`(*)`.

        Returns:
            The input tensor normalized along specified dimension(s), with shape :math:`(*)`.
        """

        v, m = torch.var_mean(x, dim=self.dim, keepdim=True)

        return (x - m) / torch.sqrt(v + self.eps)


class ModulationNd(nn.Module):
    r"""Creates an adaptive N-dimensional modulation module.

    Arguments:
        channels: The number of channels :math:`C`.
        emb_features: The number of time embedding features :math:`F`.
        spatial: The number of spatial dimensions :math:`S`.
    """

    def __init__(self, channels: int, emb_features: int, spatial: int = 2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_features, emb_features),
            nn.SiLU(),
            nn.Linear(emb_features, 3 * channels),
            Rearrange("... C -> ... C" + " 1" * spatial),
        )

        layer = self.mlp[-2]
        layer.weight = nn.Parameter(layer.weight * 1e-1)

        self.spatial = spatial

    def forward(self, t: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        r"""
        Arguments:
            t : The time embedding tensor, with shape :math:`(*, F)`.

        Returns:
            Three modulation tensors, each with shape :math:`(*, C, 1, ..., 1)`.
        """

        return torch.tensor_split(self.mlp(t), 3, dim=-(self.spatial + 1))


class PosEmbedding(nn.Module):
    r"""Creates a positional embedding module.

    References:
        | Attention Is All You Need (Vaswani et al., 2017)
        | https://arxiv.org/abs/1706.03762

    Arguments:
        features: The number of embedding features :math:`F`.
    """

    def __init__(self, features: int):
        super().__init__()
        freqs = torch.linspace(0, 1, features // 2)
        freqs = (1 / 1e4) ** freqs

        self.register_buffer("freqs", freqs)

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The position tensor, with shape :math:`(*)`.

        Returns:
            The position embedding, with shape :math:`(*, F)`.
        """

        x = x[..., None]

        return torch.cat(
            (
                torch.sin(self.freqs * x),
                torch.cos(self.freqs * x),
            ),
            dim=-1,
        )


class ResBlock(nn.Module):
    r"""Creates a convolutional residual block.

    Arguments:
        channels: The number of input/output channels of the block.
        emb_features: The number of time embedding features.
        dropout: The training dropout rate.
        spatial: The number of spatial dimensions :math:`S`.
        kwargs: Keyword arguments passed to :class:`ConvNd`.
    """

    def __init__(
        self,
        channels: int,
        emb_features: int,
        dropout: float = None,
        spatial: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.modulation = ModulationNd(channels, emb_features, spatial=spatial)
        self.block = nn.Sequential(
            LayerNorm(1),
            ConvNd(channels, channels, **kwargs),
            nn.SiLU(),
            nn.Identity() if dropout is None else nn.Dropout(dropout),
            ConvNd(channels, channels, **kwargs),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The input tensor, with shape :math:`(N, C, L_1, ..., L_S)`.
            t: The time embedding tensor, with shape :math:`(F)` or :math:`(N, F)`.

        Returns:
            The residual block output, with shape :math:`(N, C, L_1, ..., L_S)`.
        """

        a, b, c = self.modulation(t)

        y = (a + 1) * x + b
        y = self.block(y)
        y = x + c * y

        return y / torch.sqrt(1 + c**2)


class AttBlock(nn.Module):
    r"""Creates a residual self-attention block.

    Arguments:
        channels: The number of input/output channels of the block.
        emb_features: The number of time embedding features.
        heads: The number of heads of the self-attention block.
        spatial: The number of spatial dimensions :math:`S`.
    """

    def __init__(self, channels: int, emb_features: int, heads: int = 1, spatial: int = 2):
        super().__init__()
        self.modulation = ModulationNd(channels, emb_features, spatial=spatial)
        self.norm = LayerNorm(1)
        self.attn = nn.MultiheadAttention(
            num_heads=heads,
            embed_dim=channels,
            batch_first=True,
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The input tensor, with shape :math:`(N, C, L_1, ..., L_S)`.
            t: The time embedding tensor, with shape :math:`(F)` or :math:`(N, F)`.

        Returns:
            The block output, with shape :math:`(N, C, L_1, ..., L_S)`.
        """

        a, b, c = self.modulation(t)

        y = (a + 1) * x + b
        y = self.norm(y)
        y = rearrange(y, "N C ... -> N (...) C")
        y, _ = self.attn(y, y, y)
        y = rearrange(y, "N L C -> N C L").reshape(x.shape)
        y = x + c * y

        return y / torch.sqrt(1 + c**2)


class UNet(nn.Module):
    r"""Creates a time conditional UNet.

    Arguments:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        hid_channels: The number of channels of each intermediate depth.
        hid_blocks: The number of hidden blocks at each depth.
        kernel_size: The shared kernel size for all blocks.
        emb_features: The number of time embedding features.
        heads: A dictionary of pairs {depth: heads} to add self-attention at specific depth.
        dropout: The dropout rate for residual blocks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hid_channels: Sequence[int] = (64, 128, 256),
        hid_blocks: Sequence[int] = (3, 3, 3),
        kernel_size: Sequence[int] = (3, 3),
        emb_features: int = 64,
        heads: Dict[int, int] = {},  # noqa: B006
        dropout: float = None,
    ):
        super().__init__()

        assert len(hid_blocks) == len(hid_channels)

        spatial = len(kernel_size)
        stride = tuple(2 for k in kernel_size)
        kwargs = dict(
            kernel_size=kernel_size,
            padding=[k // 2 for k in kernel_size],
        )

        self.embedding = PosEmbedding(emb_features)
        self.descent, self.ascent = nn.ModuleList(), nn.ModuleList()

        for i, blocks in enumerate(hid_blocks):
            do, up = nn.ModuleList(), nn.ModuleList()

            for _ in range(blocks):
                do.append(
                    ResBlock(
                        hid_channels[i],
                        emb_features,
                        dropout=dropout,
                        spatial=spatial,
                        **kwargs,
                    )
                )
                up.append(
                    ResBlock(
                        hid_channels[i],
                        emb_features,
                        dropout=dropout,
                        spatial=spatial,
                        **kwargs,
                    )
                )

                if i in heads:
                    do.append(AttBlock(hid_channels[i], emb_features, heads[i], spatial=spatial))
                    up.append(AttBlock(hid_channels[i], emb_features, heads[i], spatial=spatial))

            if i > 0:
                do.insert(
                    0,
                    nn.Sequential(
                        ConvNd(
                            hid_channels[i - 1],
                            hid_channels[i],
                            stride=stride,
                            **kwargs,
                        ),
                        LayerNorm(1),
                    ),
                )

                up.append(
                    nn.Sequential(
                        LayerNorm(1),
                        nn.Upsample(scale_factor=stride, mode="nearest"),
                    )
                )
            else:
                do.insert(0, ConvNd(in_channels, hid_channels[i], **kwargs))
                up.append(
                    ConvNd(hid_channels[i], out_channels, kernel_size=(1,) * len(kernel_size))
                )

            if i + 1 < len(hid_blocks):
                up.insert(
                    0,
                    ConvNd(
                        hid_channels[i] + hid_channels[i + 1],
                        hid_channels[i],
                        **kwargs,
                    ),
                )

            self.descent.append(do)
            self.ascent.insert(0, up)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The input tensor, with shape :math:`(N, C, L_1, ..., L_S)`.
            t: The time tensor, with shape :math:`()` or :math:`(N)`.

        Returns:
            The output tensor, with shape :math:`(N, C, L_1, ..., L_S)`.
        """

        t = self.embedding(t)

        memory = []

        for blocks in self.descent:
            for block in blocks:
                if isinstance(block, (ResBlock, AttBlock)):
                    x = block(x, t)
                else:
                    x = block(x)

            memory.append(x)

        for blocks in self.ascent:
            y = memory.pop()
            if x is not y:
                x = torch.cat((x, y), dim=1)

            for block in blocks:
                if isinstance(block, (ResBlock, AttBlock)):
                    x = block(x, t)
                else:
                    x = block(x)

        return x
