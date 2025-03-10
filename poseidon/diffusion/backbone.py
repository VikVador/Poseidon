r"""Diffusion backbone."""

import torch.nn as nn

from einops import rearrange
from pathlib import Path
from torch import Tensor
from typing import (
    Dict,
    Sequence,
    Tuple,
)

# isort: split

from poseidon.config import PATH_MESH
from poseidon.data.const import LAND_VALUE
from poseidon.data.mask import generate_trajectory_mask
from poseidon.diffusion.tools import generate_encoded_mesh
from poseidon.network.embedding import SirenEmbedding
from poseidon.network.unet import UNet


class PoseidonBackbone(nn.Module):
    r"""Helper module used to add conditionning before denoising.

    References:
        | Elucidating the Design Space of Diffusion-Based Generative Models (Karras et al., 2022).
        | https://arxiv.org/abs/2206.00364

    Arguments:
        variables: Variable names to retain from the dataset.
        dimensions: Input tensor dimensions (B, C, K, X, Y).
        config_unet: Configuration for UNet architecture.
        config_siren: Configuration for the Siren architecture.
        config_region: Configuration for the spatial region.
    """

    def __init__(
        self,
        variables: Sequence[str],
        dimensions: Tuple[int, int, int, int, int],
        config_unet: Dict,
        config_siren: Dict,
        config_region: Dict,
        path_mesh: Path = PATH_MESH,
    ):
        super().__init__()

        self.B, self.C, self.K, self.X, self.Y = dimensions

        # Land & Sea mask
        self.register_buffer(
            "mask",
            generate_trajectory_mask(
                variables=variables,
                region=config_region,
                trajectory_size=self.K,
            ),
        )

        # Encoded mesh
        self.register_buffer(
            "mesh",
            generate_encoded_mesh(
                path=path_mesh,
                features=config_siren["features"],
                region=config_region,
            ),
        )

        # Total embedded size of the mesh
        emb_channels = self.mesh.shape[-1]
        in_channels = self.C

        # 2D UNet
        self.unet = UNet(
            in_channels=in_channels * self.K,
            out_channels=in_channels * self.K,
            blanket_size=self.K,
            **config_unet,
        )

        self.siren = SirenEmbedding(
            emb_channels,
            in_channels * self.K,  # One embedding per element of the blanket
            config_siren["n_layers"],
        )

    def forward(
        self,
        x_t: Tensor,
        sigma_t: Tensor,
    ) -> Tensor:
        r"""Denoising conditionned sample.

        Arguments:
            x_t: Noisy input tensor (B, C * K * X * Y).
            sigma_t: Associated noise levels (B, 1).

        Returns:
            Cleaned tensor (B, C * K * X * Y).
        """

        x_t = rearrange(
            x_t,
            "B (C K X Y) -> B C K X Y",
            C=self.C,
            K=self.K,
            X=self.X,
            Y=self.Y,
        )

        # Masking land
        x_t[:, self.mask[0] == 0] = LAND_VALUE

        # Adding embedded mesh
        mesh_embedding = self.siren(self.mesh)
        mesh_embedding = rearrange(
            mesh_embedding,
            "X Y (C K) -> 1 C K X Y",
            C=self.C,
            K=self.K,
            X=self.X,
            Y=self.Y,
        )

        x_t = x_t + mesh_embedding

        # Merging time and levels into channels
        x_t = rearrange(x_t, "B C K X Y -> B (C K) X Y")

        # Estimating (unscaled) clean signal
        return rearrange(
            self.unet(x_t, sigma_t),
            "B (C K) X Y -> B (C K X Y)",
            C=self.C,
            K=self.K,
            X=self.X,
            Y=self.Y,
        )
