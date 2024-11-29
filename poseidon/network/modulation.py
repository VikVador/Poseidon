r"""Modulator blocks.

Inspired by: https://github.com/probabilists/azula

"""

import torch.nn as nn

from einops.layers.torch import Rearrange
from torch import Tensor


class Modulator(nn.Module):
    r"""Generates modulating vectors for a signal given some modulating features.

    Arguments:
        channels: Number of channels in the target signal.
        mod_features: Number of features (D) in the modulating vector (B, D).
        spatial: Number of spatial dimensions (used for convolution) in the target signal.

    Returns:
        Tensor of modulating vectors with shape (3, B, C, SPATIAL)
    """

    def __init__(self, channels: int, mod_features: int, spatial: int):
        super().__init__()

        self.ada_zero = nn.Sequential(
            nn.Linear(mod_features, mod_features),
            nn.SiLU(),
            nn.Linear(mod_features, 3 * channels),
            Rearrange("... (r C) -> r ... C" + " 1" * spatial, r=3),
        )

        # Downscaling weights for the modulation generator
        layer = self.ada_zero[-2]
        layer.weight = nn.Parameter(layer.weight * 1e-2)

    def forward(self, mod: Tensor) -> Tensor:
        r"""Computes modulating vectors for the input tensor.

        Arguments:
            mod: Input tensor of shape (B, D), where B is the batch size and D is the number of modulating features.

        Returns:
            Tensor: Modulating vectors with shape (3, B, C, ...), where `...` corresponds to the spatial dimensions.
        """
        return self.ada_zero(mod)
