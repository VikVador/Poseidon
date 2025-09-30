r"""U-Net building blocks.

Credits:
    https://github.com/probabilists/azula
"""

import torch
import torch.nn as nn

from einops import repeat
from einops.layers.torch import Rearrange
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from typing import Dict, Optional, Sequence, Union

# isort: split
from poseidon.network.unet.encoding import SineEncoding
from poseidon.network.unet.layers import ConvNd, LayerNorm, SelfAttentionNd


class UNetBlock(nn.Module):
    r"""Creates a modulated U-Net block module.

    Arguments:
        channels: Number of channels (C).
        mod_features: Number of modulating features (D).
        norm: Kind of normalization.
        groups: Number of groups in torch.nn.GroupNorm layers.
        attention_heads: Number of attention heads.
        ffn_factor: Channel factor in the FFN.
        spatial: Number of spatial dimensions (N).
        dropout: Dropout rate in [0, 1].
        checkpointing: Whether to use gradient checkpointing or not.
        kwargs: Keyword arguments passed to torch.nn.Conv2d.
    """

    def __init__(
        self,
        channels: int,
        mod_features: int = 0,
        norm: str = "layer",
        groups: int = 16,
        attention_heads: Optional[int] = None,
        ffn_factor: int = 1,
        spatial: int = 2,
        dropout: Optional[float] = None,
        checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.checkpointing = checkpointing

        # Attention
        if attention_heads is None:
            self.attn = None
        else:
            self.attn = SelfAttentionNd(channels, heads=attention_heads)

            kwargs.update(kernel_size=1, padding=0)

        # Ada-Norm Zero
        if norm == "layer":
            self.norm = LayerNorm(dim=-spatial - 1)
        elif norm == "group":
            self.norm = nn.GroupNorm(
                num_groups=min(groups, channels),
                num_channels=channels,
                affine=False,
            )
        else:
            raise NotImplementedError()

        if mod_features > 0:
            self.ada_zero = nn.Sequential(
                nn.Linear(mod_features, mod_features),
                nn.SiLU(),
                nn.Linear(mod_features, 3 * channels),
                Rearrange("... (n C) -> n ... C" + " 1" * spatial, n=3),
            )

            self.ada_zero[-2].weight.data.mul_(1e-2)
        else:
            self.ada_zero = nn.Parameter(torch.randn(3, channels, *(1,) * spatial))
            self.ada_zero.data.mul_(1e-2)

        # Block
        self.ffn = nn.Sequential(
            ConvNd(channels, ffn_factor * channels, spatial=spatial, **kwargs),
            nn.SiLU(),
            nn.Identity() if dropout is None else nn.Dropout(dropout),
            ConvNd(ffn_factor * channels, channels, spatial=spatial, **kwargs),
        )

    def _forward(self, x: Tensor, mod: Optional[Tensor] = None) -> Tensor:
        r"""
        Arguments:
            x: Input tensor (B, C, L_1, ..., L_N).
            mod: Modulation vector (D) or (B, D).

        Returns:
            output: Tensor (B, C, L_1, ..., L_N).
        """

        if torch.is_tensor(self.ada_zero):
            a, b, c = self.ada_zero
        else:
            a, b, c = self.ada_zero(mod)

        y = (a + 1) * self.norm(x) + b
        y = y if self.attn is None else y + self.attn(y)
        y = self.ffn(y)
        y = (x + c * y) * torch.rsqrt(1 + c * c)

        return y

    def forward(
        self,
        x: Tensor,
        mod: Optional[Tensor] = None,
    ) -> Tensor:
        if self.checkpointing:
            return checkpoint(self._forward, x, mod, use_reentrant=False)
        else:
            return self._forward(x, mod)


class UNet(nn.Module):
    r"""Creates a modulated U-Net module.

    Information:
        Conditioning: Additional information concatenated to the input.
        Modulation: Additional information used to modulate the blocks.

    Arguments:
        in_channels: Number of input channels (C_i).
        out_channels: Number of output channels (C_o).
        cond_channels: Number of condition channels (C_c).
        mod_features: Number of modulating features (D).
        hid_channels: Numbers of channels at each depth.
        hid_blocks: Numbers of hidden blocks at each depth.
        kernel_size: Kernel size of all convolutions.
        stride: Stride of the downsampling convolutions.
        norm: Type of normalization.
        attention_heads: Number of attention heads at each depth.
        spatial: Number of spatial dimensions (N).
        periodic: Whether the spatial dimensions are periodic or not.
        dropout: Dropout rate in [0, 1].
        checkpointing: Whether to use gradient checkpointing or not.
        identity_init: Initialize down/upsampling convolutions as identity.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int = 1,
        mod_features: int = 1,
        hid_channels: Sequence[int] = (16, 32, 64),
        hid_blocks: Sequence[int] = (1, 1, 1),
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 2,
        norm: str = "layer",
        attention_heads: Dict[int, int] = {},  # noqa: B006
        ffn_factor: int = 1,
        spatial: int = 3,
        periodic: bool = False,
        dropout: Optional[float] = None,
        checkpointing: bool = True,
        identity_init: bool = False,
    ):
        super().__init__()

        assert len(hid_blocks) == len(hid_channels)

        if isinstance(kernel_size, int):
            kernel_size = [1, kernel_size, kernel_size]

        if isinstance(stride, int):
            stride = [1, stride, stride]

        self.modulation_encoding = SineEncoding(mod_features)

        self.descent, self.ascent = nn.ModuleList(), nn.ModuleList()

        kwargs = dict(
            kernel_size=tuple(kernel_size),
            padding=tuple(k // 2 for k in kernel_size),
            padding_mode="circular" if periodic else "zeros",
        )

        for i, num_blocks in enumerate(hid_blocks):
            do, up = nn.ModuleList(), nn.ModuleList()

            # Stage blocks
            for _ in range(num_blocks):
                do.append(
                    UNetBlock(
                        hid_channels[i],
                        mod_features,
                        norm=norm,
                        attention_heads=attention_heads.get(i, None),
                        ffn_factor=ffn_factor,
                        spatial=spatial,
                        dropout=dropout,
                        checkpointing=checkpointing,
                        **kwargs,
                    )
                )

                up.append(
                    UNetBlock(
                        hid_channels[i],
                        mod_features,
                        norm=norm,
                        attention_heads=attention_heads.get(i, None),
                        ffn_factor=ffn_factor,
                        spatial=spatial,
                        dropout=dropout,
                        checkpointing=checkpointing,
                        **kwargs,
                    )
                )

            # Transitions
            if i > 0:
                do.insert(
                    0,
                    ConvNd(
                        hid_channels[i - 1],
                        hid_channels[i],
                        stride=stride,
                        spatial=spatial,
                        identity_init=identity_init,
                        **kwargs,
                    ),
                )

                up.append(
                    nn.Upsample(
                        scale_factor=tuple(stride),
                        mode="nearest",
                    )
                )

            # Initial / Final
            else:
                do.insert(
                    0,
                    ConvNd(
                        in_channels + cond_channels + spatial,
                        hid_channels[i],
                        spatial=spatial,
                        **kwargs,
                    ),
                )
                up.append(
                    ConvNd(
                        hid_channels[i],
                        out_channels,
                        spatial=spatial,
                        **kwargs,
                    )
                )

            # Merging skip connections
            if i + 1 < len(hid_blocks):
                up.insert(
                    0,
                    ConvNd(
                        hid_channels[i] + hid_channels[i + 1],
                        hid_channels[i],
                        spatial=spatial,
                        identity_init=identity_init,
                        **kwargs,
                    ),
                )

            self.descent.append(do)
            self.ascent.insert(0, up)

    def forward(
        self,
        x: Tensor,
        mod: Optional[Tensor] = None,
        cond: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Arguments:
            x: Input tensor (B, C_i, L_1, ..., L_N).
            mod: Modulation vector (D) or (B, D).
            cond: Condition tensor (B, E).

        Returns:
            tensor: (B, C_o, L_1, ..., L_N).
        """

        B, _, *shape = x.shape

        # Creating spatial positional encodings
        p = (torch.linspace(-1, 1, steps=size, device=x.device) for size in shape)
        p = torch.cartesian_prod(*p)
        p = torch.reshape(p, shape=(*shape, -1))
        p = repeat(p, "... C -> B C ...", B=B)

        # Projecting conditioning to spatial dimensions
        cond = (
            torch.ones(
                B,
                1,
                *shape,
                device=x.device,
            )
            * cond[:, :, None, None, None]
        )

        # Encoding modulating vector
        mod = self.modulation_encoding(mod)[:, 0].to(x.device)

        # Adding information to the input
        if cond is None:
            x = torch.cat((x, p), dim=1)
        else:
            x = torch.cat((x, p, cond), dim=1)

        # Forward through the network
        memory = []

        for blocks in self.descent:
            for block in blocks:
                if isinstance(block, UNetBlock):
                    x = block(x, mod)
                else:
                    x = block(x)

            memory.append(x)

        for blocks in self.ascent:
            y = memory.pop()
            if x is not y:
                for i in range(2, x.ndim):
                    if x.shape[i] > y.shape[i]:
                        x = torch.narrow(x, i, 0, y.shape[i])

                x = torch.cat((x, y), dim=1)

            for block in blocks:
                if isinstance(block, UNetBlock):
                    x = block(x, mod)
                else:
                    x = block(x)

        return x
