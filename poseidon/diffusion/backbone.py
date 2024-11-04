r"""Diffusion - Backbone module helping preprocessing data conditionning."""

import torch
import torch.nn as nn
import xarray as xr

from einops import rearrange
from typing import Dict, Tuple

# isort: split
from poseidon.config import (
    POSEIDON_MASK,
    POSEIDON_MESH,
)
from poseidon.network.unet import UNet


def generate_mesh_mask(region: Dict) -> torch.Tensor:
    r"""Load the Black Sea mesh and mask to create a spatial conditioning.

    Arguments:
        region: Contains information about the region of interest.

    Returns:
        Normalized spatial information with shape (4, level, lattitude, longitude).
    """
    mask = xr.open_zarr(POSEIDON_MASK).sel(**region).mask.values
    mesh = xr.open_zarr(POSEIDON_MESH).sel(**region)

    # Convert normalized mesh data (x, y, z) to PyTorch tensors
    mx = torch.from_numpy(mesh.x_mesh.values) / (region["longitude"].stop - 1)
    my = torch.from_numpy(mesh.y_mesh.values) / (region["latitude"].stop - 1)
    mz = torch.from_numpy(mesh.z_mesh.values) / (region["level"].stop - 1)
    mesh = torch.cat([mx.unsqueeze(0), my.unsqueeze(0), mz.unsqueeze(0)], dim=0)
    return torch.cat([torch.from_numpy(mask).unsqueeze(0), mesh], dim=0)


class PoseidonBackbone(nn.Module):
    r"""A helper module used to preprocess data for the Poseidon neural network.

    It is responsible for preparing the input data and temporal embeddings to condition
    the neural network, based on a spatial mask, time embeddings (hour, day, month),
    and a neighborhood blanket size.

    References:
        | Elucidating the Design Space of Diffusion-Based Generative Models (Karras et al., 2022).
        | https://arxiv.org/abs/2206.00364

    Arguments:
        k: Number of neighbors to consider on each side to define the "blanket"
        dimensions: Contains number of channels, latitude, and longitude of the data
        config_nn: Configuration parameters for the UNet neural network architecture
        config_region: Configuration for the spatial region.
    """

    def __init__(
        self, k: int, dimensions: Tuple[int, int, int, int], config_nn: Dict, config_region: Dict
    ):
        super().__init__()
        self.k = k
        self.blanket_size = k * 2 + 1
        self.channels = dimensions[0]
        self.traj_size = dimensions[1]
        self.latitude = dimensions[2]
        self.longitude = dimensions[3]
        nb_pixels = self.latitude * self.longitude
        self.register_buffer(
            "spatial", generate_mesh_mask(config_region).flatten(start_dim=0, end_dim=1)
        )
        self.embedding_hour = nn.Embedding(24, nb_pixels)
        self.embedding_day = nn.Embedding(31, nb_pixels)
        self.embedding_month = nn.Embedding(12, nb_pixels)
        # Composed of: Blanket * Channels + Time (3) + Spatial (Mesh x,y,z and mask)(4)
        self.in_channels = (
            self.channels
            + 3
            + 4 * config_region["level"].stop  # Total number of levels in dataset
        )
        self.out_channels = self.channels
        self.neural_network = UNet(
            in_channels=self.in_channels, out_channels=self.out_channels, **config_nn
        )

    def forward(self, x: torch.Tensor, sigma: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        r"""Performs a forward pass through the network.

        Arguments:
            x: Noisy input tensor with shape (B, D).
            sigma: Noise level of a specific diffusion step (B, 1).
            c: Tokenized time-based conditioning (B, 3), i.e. month, day, hour.

        Returns:
            Denoised tensor (B, D).
        """

        # Mixing blanket and variables
        x = rearrange(
            x,
            "B (C K H W) -> B C K H W",
            C=self.channels,
            K=self.blanket_size,
            H=self.latitude,
            W=self.longitude,
        )

        # Embedding temporal conditioning
        embd_month, embd_day, embd_hour = (
            self.embedding_month(c[:, 0]),
            self.embedding_day(c[:, 1]),
            self.embedding_hour(c[:, 2]),
        )
        embd_month, embd_day, embd_hour = (
            rearrange(embd_month, "B (C K H W) -> B C K H W", C=1,K=1, H=self.latitude, W=self.longitude),
            rearrange(embd_day, "B (C K H W) -> B C K H W", C=1,K=1, H=self.latitude, W=self.longitude),
            rearrange(embd_hour, "B (C K H W) -> B C K H W", C=1,K=1, H=self.latitude, W=self.longitude),
        )

        # Broadcasting the temporal mask to the blanket size
        embd_month, embd_day, embd_hour = (
            embd_month.repeat(1, 1, self.blanket_size, 1, 1),
            embd_day.repeat(1, 1, self.blanket_size, 1, 1),
            embd_hour.repeat(1, 1, self.blanket_size, 1, 1),
        )

        # Broadcasting the spatial mask to batch size
        embd_spatial = self.spatial.repeat(x.shape[0], self.blanket_size, 1, 1, 1)
        embd_spatial = rearrange(embd_spatial, "B K C H W -> B C K H W")

        # Conditioning the input (converting mask to float)
        x = torch.cat([x, embd_spatial, embd_month, embd_day, embd_hour], dim=1).float()

        # Forward pass ('t' used to be diffusion step, now it's the noise level)
        x = self.neural_network(x=x, t=sigma)
        x = rearrange(x, "B ... -> B (...)")
        return x
