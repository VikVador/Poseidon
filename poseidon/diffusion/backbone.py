r"""Diffusion backbone helping conditionning data."""

import torch.nn as nn

from einops import rearrange
from pathlib import Path
from torch import Tensor
from typing import Dict, Tuple

# isort: split
from poseidon.config import PATH_MESH
from poseidon.diffusion.tools import generate_encoded_mesh
from poseidon.network.embedding import SirenEmbedding
from poseidon.network.unet import UNet


class PoseidonBackbone(nn.Module):
    r"""A diffusion helper used to add data conditionning.

    Information:
        Responsible for preparing spatial embeddings to add context to the input data.

    Assumptions:
        Simulator dynamics, P(x_{t+1} | x_t), is time-independent.

    References:
        | Elucidating the Design Space of Diffusion-Based Generative Models (Karras et al., 2022).
        | https://arxiv.org/abs/2206.00364

    Arguments:
        dimensions: Input tensor dimensions (B, C, K, H, W).
        config_unet: Configuration the UNet architecture.
        config_siren: Configuration for the Siren architecture.
        config_region: Configuration for the spatial region.
    """

    def __init__(
        self,
        dimensions: Tuple[int, int, int, int, int],
        config_unet: Dict,
        config_siren: Dict,
        config_region: Dict,
        path_mesh: Path = PATH_MESH,
    ):
        super().__init__()

        self.B, self.C, self.K, self.H, self.W = dimensions

        # Sin/cos encoded mesh
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

        self.unet = UNet(
            in_channels,
            in_channels,
            blanket_size=self.K,
            **config_unet,
        )

        self.siren = SirenEmbedding(
            emb_channels,
            in_channels * self.K,  # One embedding per element of the blanket
            config_siren["n_layers"],
        )

    def forward(self, x: Tensor, sigma: Tensor) -> Tensor:
        r"""Condition the input and denoise it by forwarding it through the UNet.

        Arguments:
            x: Noisy input tensor (B, D).
            sigma: Noise level of diffusion step (B, 1).

        Returns:
            Denoised tensor (B, D).
        """

        x = rearrange(
            x,
            "B (C K H W) -> B C K H W",
            C=self.C,
            K=self.K,
            H=self.H,
            W=self.W,
        )

        # Adding embedded mesh to the input
        mesh_embedding = self.siren(self.mesh)

        mesh_embedding = rearrange(
            mesh_embedding,
            "H W (C K) -> 1 C K H W",
            C=self.C,
            K=self.K,
            H=self.H,
            W=self.W,
        )

        x = x + mesh_embedding

        # Denoising
        x = self.unet(x, sigma)

        return rearrange(x, "B ... -> B (...)")
