r"""Score of the prior distribution."""

import torch
import torch.nn as nn

from einops import rearrange
from torch import Tensor

# isort: split
from poseidon.diffusion.denoiser import PoseidonDenoiser
from poseidon.training.tools import (
    compute_blanket_indices,
)

# Constants
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_LIST = [i for i in range(torch.cuda.device_count())]


class PoseidonScorePrior(nn.Module):
    r"""Computes the score of the prior distribution using a pre-trained :class:`PoseidonDenoiser`.

    Mathematics:
        ∇_x_{1:T} log p(x_{1:T} ; σ) = [ Denoiser(x_{1:T}, σ) - x_{1:T} ] / σ^2

    Arguments:
        denoiser: A trained :class:`PoseidonDenoiser` model.
        parallelize: Wether or not parallelize the forward pass of the denoiser.
    """

    def __init__(
        self,
        denoiser: PoseidonDenoiser,
        parallelize: bool = False,
    ):
        super().__init__()

        self.blanket_neighbors, self.blanket_size = (
            denoiser.backbone.K // 2,
            denoiser.backbone.K,
        )

        # Parallelizing is needed only for long trajectories prediction
        self.denoiser = self._parallelize(denoiser).to(DEVICE) if parallelize else denoiser

    def forward(self, x: Tensor, sigma: Tensor) -> Tensor:
        r"""Compute the score of prior distribution of a trajectory.

        Arguments:
            x: Trajectory (C, T, H, W).
            sigma: Noise (T, 1).

        Returns:
            Score of the prior distribution (C, T, H, W).
        """

        # Extracting dimensions on the fly (allows dynamic region size)
        self.C, _, self.H, self.W = x.shape

        # Creating blankets for each state of the trajectory
        x = self._create_blankets(x)

        # Rearranging the tensor for the denoiser
        x = rearrange(
            x,
            "T C K H W -> T (C K H W)",
        )

        # Computes the score of each blanket
        x = (self.denoiser(x, sigma) - x) / sigma**2

        # Extracting back original structure
        x = rearrange(
            x,
            "T (C K H W) -> T C K H W",
            C=self.C,
            K=self.blanket_size,
            H=self.H,
            W=self.W,
        )

        # Determine the score of the trajectory
        return self._extract_score_trajectory(x)

    def _parallelize(self, denoiser: PoseidonDenoiser):
        r"""Parallelize the forward pass of the denoiser model."""
        return (
            nn.DataParallel(denoiser, device_ids=DEVICE_LIST)
            if torch.cuda.device_count() > 1
            else denoiser
        )

    def _create_blankets(self, x: Tensor) -> Tensor:
        r"""Creates blankets of size (K) for each state of a trajectory.

        Arguments:
            x: Trajectory (C, T, H, W).

        Returns:
            Trajectory decomposed as blankets (T, C, K, H, W).
        """

        # Extracting dynamically trajectory size
        _, trajectory_size, _, _ = x.shape

        # Determine the position of each blanket, extract and stack them
        return torch.stack(
            [
                x[:, idx[0] : idx[1]]
                for _, idx in compute_blanket_indices(
                    trajectory_size=trajectory_size,
                    k=self.blanket_neighbors,
                ).items()
            ],
            dim=0,
        )

    def _extract_score_trajectory(self, score_blankets: Tensor):
        r"""Extracts the score of trajectory states in the blankets.

        Arguments:
            score_blankets: Scores for each blanket of size (K) (T, C, K, H, W).

        Returns:
            Score of trajectory (C, T, H, W).
        """

        # Extracting dynamically trajectory size
        trajectory_size, _, _, _, _ = score_blankets.shape

        # ==== MAGIC =====
        # In each blanket, this trick determine the index of the corresponding
        # trajectory state (it's a complicated business at the edges !)
        #
        trajectory_state_indices = (
            [i for i in range(self.blanket_neighbors)]
            + [self.blanket_neighbors for _ in range(trajectory_size - 2 * self.blanket_neighbors)]
            + [(2 * self.blanket_neighbors + 1) - i for i in range(self.blanket_neighbors, 0, -1)]
        )

        # Extract the score of the trajectory
        return torch.stack(
            [score_blankets[i, :, idx] for i, idx in enumerate(trajectory_state_indices)],
            dim=1,
        )
