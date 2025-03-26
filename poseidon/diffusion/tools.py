r"""A collection of tools designed for diffusion module."""

import torch
import torch.nn as nn
import xarray as xr

from einops import rearrange
from pathlib import Path
from torch import Tensor
from typing import Dict, Tuple

# isort: split
from poseidon.network.encoding import SineEncoding


def generate_encoded_mesh(
    path: Path,
    features: int,
    region: Dict,
) -> Tensor:
    """Generates a sin/cos encoded mesh of a Black Sea region.

    Arguments:
        path: Path to the Black Sea mesh.
        features: Even number of sin/cos embeddings (F).
        region: Region of interest to extract from the dataset.

    Returns:
        Tensor: Encoded mesh (X Y (Mesh Levels F)).
    """

    mesh_data = xr.open_zarr(path).isel(**region).load()

    # Stack mesh variables into a single tensor
    mesh = torch.stack(
        [torch.from_numpy(mesh_data[v].values) for v in mesh_data.variables],
        dim=0,
    )
    mesh = rearrange(
        SineEncoding(features).forward(mesh),
        "... X Y F -> X Y (F ...)",
    )

    return mesh.to(dtype=torch.float32)


class PoseidonTrajectoryWrapper(nn.Module):
    r"""Wrap a blanket denoising model into a full trajectory model.

    Arguments:
        denoiser: A denoiser model d(xₜ) ≈ E[x | xₜ]
        dimensions: Dimensions of the trajectory (C, X, Y).
        blanket_size: Dimension of the blanket.
    """

    def __init__(
        self,
        denoiser: nn.Module,
        dimensions: Tuple[int, int, int],
        blanket_size: int,
    ):
        super().__init__()

        self.C, self.X, self.Y = dimensions

        self.denoiser, self.blanket_neighbors, self.blanket_size = (
            denoiser,
            (blanket_size // 2),
            blanket_size,
        )

    def forward(self, x_t: Tensor, sigma_t: Tensor) -> Tensor:
        r"""Denoises a trajectory using a denoiser.

        Arguments:
            x_t: Noisy trajectory (C, T, X, Y).
            sigma_t: Noise (T, 1).

        Returns:
            Denoised trajectory (C, T, X, Y).
        """

        # Creating blankets for each state of the trajectory
        x_t = self._create_blankets(x_t)

        # Rearranging the tensor for the denoiser
        x_t = rearrange(
            x_t,
            "B C K X Y -> B (C K X Y)",
        )

        # Recreating sigma_t for each blanket
        sigma_t = sigma_t * torch.ones(x_t.shape[0], 1).to(x_t.device)

        # Denoising the blankets
        x_t = self.denoiser(x_t, sigma_t)

        # Extracting back original structure
        x_t = rearrange(
            x_t,
            "B (C K X Y) -> B C K X Y",
            C=self.C,
            K=self.blanket_size,
            X=self.X,
            Y=self.Y,
        )

        # Determine the score of the trajectory
        return self._extract_states(x_t)

    def _create_blankets(self, x_t: Tensor) -> Tensor:
        r"""Creates blankets of size (K) from a trajectory.

        Information:
            States at the edges are contained in only one blanket (per side).

        Arguments:
            x_t: Noisy trajectory (C, T, X, Y).

        Returns:
            Trajectory decomposed efficiently into blankets (B, C, K, X, Y).
        """

        # Updating trajectory size
        self.trajectory_size = x_t.shape[1]

        # Creating blankets
        x_t = x_t.unfold(dimension=1, size=self.blanket_size, step=1)
        x_t = rearrange(x_t, "C B X Y K -> B C K X Y")

        return x_t

    def _extract_states(self, x_t: Tensor) -> Tensor:
        r"""Recompose trajectory from blankets.

        Arguments:
            x_t: Denoised blankets (B, C, K, X, Y).

        Returns:
            Trajectory (C, T, X, Y).
        """

        # Extracting dimensions
        B, _, K, _, _ = x_t.shape

        # Case 1 - One blanket to cover trajectory
        if B == 1:
            return x_t[0]

        # Case 2 - Two blankets to cover trajectory
        elif B == 2:
            return torch.concat(
                [x_t[0, :, : self.blanket_size], x_t[1, :, -(self.trajectory_size - K) :]], dim=1
            )

        # Case 3 - Multiple blankets to cover whole trajectory
        else:
            idx_start = self.blanket_size - self.blanket_neighbors

            x_start = x_t[0, :, :idx_start]
            x_end = x_t[-1, :, -idx_start:]
            x_middle = torch.cat(
                [x_t[i, :, self.blanket_size // 2].unsqueeze(1) for i in range(1, B - 1)], dim=1
            )

            return torch.cat([x_start, x_middle, x_end], dim=1)
