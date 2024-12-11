r"""Diffusion backbone helping conditionning data."""

import torch
import torch.nn as nn
import xarray as xr

from einops import rearrange
from poseidon.config import PATH_MESH
from typing import Dict, Tuple

# isort: split
from poseidon.network.embedding import SirenEmbedding
from poseidon.network.encoding import SineEncoding
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
        config_region: Configuration for the spatial region used for training.
    """

    def __init__(
        self,
        dimensions: Tuple[int, int, int, int, int],
        config_unet: Dict,
        config_siren: Dict,
        config_region: Dict,
        device: str = "cpu",
    ):
        super().__init__()

        # Initialization
        self.region = config_region
        self.B, self.C, self.K, self.H, self.W = dimensions
        self.mesh = self.generate_encoded_mesh(
            features=config_siren["features"],
        ).to(device)

        # Total embedded size of the mesh
        emb_channels = self.mesh.shape[-1]
        in_channels = self.C

        self.unet = UNet(
            in_channels,
            in_channels,
            **config_unet,
        ).to(device)

        self.siren = SirenEmbedding(
            emb_channels,
            in_channels * self.K,  # One embedding per element of the blanket
            config_siren["n_layers"],
        ).to(device)

    def generate_encoded_mesh(self, features: int) -> torch.Tensor:
        """Load the Black Sea mesh and apply a sine encoding.

        Arguments:
            features: Number of embedding features (F). Must be even.

        Returns:
            Tensor: Encoded mesh tensor  X Y (Mesh Levels Features).
        """

        mesh_data = xr.open_zarr(PATH_MESH).isel(**self.region).load()

        # Stack mesh variables into a single tensor
        mesh = torch.stack(
            [torch.from_numpy(mesh_data[v].values) for v in mesh_data.variables],
            dim=0,
        )
        mesh = rearrange(
            SineEncoding(features).forward(mesh),
            "... X Y F -> X Y (F ...)",
        )
        return mesh

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
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
