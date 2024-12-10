r"""Modulator blocks."""

import torch.nn as nn

from einops.layers.torch import Rearrange
from torch import Tensor


class Modulator(nn.Module):
    r"""Generates modulating vectors for a signal given some modulating features.

    Arguments:
        channels: Number of channels (C) in the target signal (B, C, *).
        mod_features: Number of features (D) in the modulating vector (B, D).
        spatial: Number of spatial dimensions (used for convolution) in the target signal.
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
            mod: Modulating vector (B, D).

        Returns:
            Tensor: Modulating vectors with shape (3, B, C, SPATIAL)
        """
        return self.ada_zero(mod)
