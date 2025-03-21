r"""Embedding blocks."""

import math
import torch
import torch.nn as nn

from einops import rearrange
from torch import Tensor
from typing import Dict

# isort: split
from poseidon.config import PATH_MESH
from poseidon.diffusion.tools import generate_encoded_mesh


class SineLayer(nn.Module):
    r"""Adapted implementation of a SineLayer for SirenNet.

    Reference:
        | https://github.com/vsitzmann/siren

    Arguments:
        in_features: Number of input features (*, I).
        out_features: Number of output features (*, O).
        is_first: Whether the layer is the first of :class:`SirenNet` or not.
        omega_0: Boosting factor of the layer (described in supplement Sec. 1.5 or original paper).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        is_first: bool = False,
        omega_0: float = 30.0,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights(is_first)

    @torch.no_grad()
    def init_weights(self, is_first) -> None:
        if is_first:
            weight_bound = 1 / self.in_features
        else:
            weight_bound = math.sqrt(6 / self.in_features) / self.omega_0**2

        self.linear.weight.uniform_(-weight_bound, weight_bound)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SirenEmbedding(nn.Module):
    r"""Creates a positional embedding module based on SirenNet.

    Reference:
        | Implicit Neural Representations with Periodic Activation Functions
        | https://arxiv.org/abs/2006.09661
        | Geographic Location Encoding with Spherical Harmonics and Sinusoidal Representation Networks
        | https://arxiv.org/abs/2310.06743

    Arguments:
        in_features: Number of input features (*, I).
        out_features: Number of output features (*, O).
        n_layers: Number of hidden SineLayers.
        omega_0: Boosting factor of the layer (described in supplement Sec. 1.5 or original paper).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_layers: int,
        omega_0: float = 30.0,
    ):
        super().__init__()
        layers = []
        layers.append(
            SineLayer(
                in_features,
                out_features,
                is_first=True,
                omega_0=omega_0,
            )
        )
        layers.extend([
            SineLayer(
                out_features,
                out_features,
                is_first=False,
                omega_0=omega_0,
            )
            for _ in range(n_layers)
        ])
        layers.append(nn.Linear(out_features, out_features))
        self.siren = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.siren(x)


class MeshEmbedding(nn.Module):
    r"""Creates a block which generates a spatial mesh embedding.

    Arguments:
        channels: Number of channels on which projecting the spatial mesh.
        features: Number of features to encode the spatial mesh.
        n_layers: Number of layers for the Siren embedding.
        spatial_scaling: Upscaling factor for the spatial mesh.
        config_region: Configuration for the region.
    """

    def __init__(
        self,
        channels: int,
        features: int,
        n_layers: int,
        spatial_scaling: int,
        config_region: Dict,
    ):
        super().__init__()

        # Creating the spatial mesh
        mesh = generate_encoded_mesh(
            path=PATH_MESH,
            features=features,
            region=config_region,
        )

        # Extracting dimensions
        X, Y, _ = mesh.shape

        # Upscaling the mesh (to math UNet stage resolution)
        self.register_buffer(
            "mesh",
            mesh[: X // (2**spatial_scaling), : Y // (2**spatial_scaling), :],
        )

        # Embedding layer
        self.siren = SirenEmbedding(
            in_features=mesh.shape[-1],
            out_features=channels,
            n_layers=n_layers,
        )

    def forward(self) -> Tensor:
        r"""Generates a spatial mesh embedding."""

        # Embedding the spatial mesh
        mesh = self.siren(self.mesh)

        # Rearranging for broadcasting
        return rearrange(mesh, "X Y C -> 1 C X Y")
