r"""UDiT Architecture."""

import torch
import torch.nn as nn

from torch import Tensor
from typing import Dict, Sequence

# isort: split
from poseidon.network.convolution import ConvNd, UNetBlock
from poseidon.network.transformer import Transformer


# fmt: off
#
class UDiT(nn.Module):
    r"""Creates a U-Net diffusion transformer.

    Arguments:
        in_channels: Number of input channels (C_i)
        out_channels: Number of output channels (C_o).
        kernel_size: Kernel size for spatial convolutions.
        blanket_size: Total number of elements in a blanket (K).
        mod_features: Number of features (D) in modulating vector.
        ffn_scaling: Scaling factor for the feed-forward network.
        hid_channels: Numbers of channels at each depth.
        hid_blocks: Numbers of hidden blocks at each depth.
        attention_heads: The number of attention heads at each depth.
        config_siren: Configuration of embedding mesh.
        config_region: Configuration of spatial region.
        config_transformer: Configuration for the Transformer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        blanket_size: int,
        mod_features: int,
        ffn_scaling: int,
        hid_blocks: Sequence[int],
        hid_channels: Sequence[int],
        attention_heads: Dict[str, int],
        config_siren: Dict,
        config_region: Dict,
        config_transformer: Dict,
    ):
        super().__init__()

        # Security
        assert len(hid_blocks) == len(hid_channels), \
            "ERROR (UDit) - Mismatched number of hidden blocks and channels."

        assert kernel_size % 2 == 1, \
            "ERROR (UDit) - Kernel size must be odd."

        # Transformer bottleneck
        self.transformer = Transformer(
            in_channels=hid_channels[-1] * config_transformer["patch_size"] ** 2 * blanket_size,
            mod_features=mod_features,
            **config_transformer,
        )

        # Updating to handle concatenated channels projection
        hid_channels = hid_channels + [hid_channels[-1]]

        # Convolutional residual blocks
        kwargs = dict(
            kernel_size=(
                3,
                kernel_size,
                kernel_size,
            ),
            padding=(
                3 // 2,
                kernel_size // 2,
                kernel_size // 2,
            ),
        )

        # Contains the blocks for the descent and ascent of the UNet
        self.descent, self.ascent = nn.ModuleList(), nn.ModuleList()

        for i, blocks in enumerate(hid_blocks):
            do, up = nn.ModuleList(), nn.ModuleList()

            for _ in range(blocks):
                do.append(
                    UNetBlock(
                        hid_channels[i],
                        mod_features,
                        ffn_scaling,
                        i,
                        config_region,
                        config_siren,
                        attention_heads.get(str(i), None),
                        **kwargs,
                    )
                )
                up.append(
                    UNetBlock(
                        hid_channels[i],
                        mod_features,
                        ffn_scaling,
                        i,
                        config_region,
                        config_siren,
                        attention_heads.get(str(i), None),
                        **kwargs,
                    )
                )

            if i > 0:
                do.insert(
                    index=0,
                    module=nn.Sequential(
                        ConvNd(
                            hid_channels[i - 1],
                            hid_channels[i],
                            spatial=3,
                            stride=(1, 2, 2),
                            **kwargs,
                        ),
                    ),
                )

                up.append(
                    nn.Sequential(
                        nn.Upsample(
                            scale_factor=(1, 2, 2),
                            mode="nearest",
                        ),
                    )
                )

            else:
                do.insert(
                    0,
                    ConvNd(
                        in_channels,
                        hid_channels[i],
                        spatial=3,
                        **kwargs,
                    ),
                )
                up.append(
                    ConvNd(
                        hid_channels[i],
                        out_channels,
                        spatial=3,
                        **kwargs,
                    )
                )

            if i + 1 <= len(hid_blocks):
                up.insert(
                    0,
                    ConvNd(
                        hid_channels[i] + hid_channels[i + 1],
                        hid_channels[i],
                        spatial=3,
                        **kwargs,
                    ),
                )

            self.descent.append(do)
            self.ascent.insert(0, up)

    def forward(self, x: Tensor, mod: Tensor) -> Tensor:
        r"""Forward pass through the UDiT."""

        # Stores output of each ascent stage
        memory = []

        # Ascent
        for blocks in self.descent:
            for block in blocks:
                if isinstance(block, (UNetBlock)):
                    x = block(x, mod)
                else:
                    x = block(x)
            memory.append(x)

        # Transformer bottleneck
        x = self.transformer(x, mod)

        # Descent
        for blocks in self.ascent:
            y = memory.pop()
            if x is not y:
                for i in range(2, x.ndim):
                    if x.shape[i] > y.shape[i]:
                        x = torch.narrow(x, i, 0, y.shape[i])
                x = torch.cat((x, y), dim=1)
            for block in blocks:
                if isinstance(block, (UNetBlock)):
                    x = block(x, mod)
                else:
                    x = block(x)

        return x
