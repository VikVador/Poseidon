r"""Diffusion wrappers."""

import torch
import torch.nn as nn

from einops import rearrange
from torch import Tensor
from typing import Tuple


class PoseidonTrajectoryWrapper(nn.Module):
    r"""Transform blanket denoiser into trajectory denoiser.

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

    def forward(
        self,
        x_t: Tensor,
        sigma_t: Tensor,
        cond: Tensor,
    ) -> Tensor:
        r"""Denoises a trajectory using a denoiser.

        Arguments:
            x_t: Noisy trajectory (C, T, X, Y).
            sigma_t: Associated noise levels (B, 1).
            cond: Associated conditioning tensor (B, K).

        Returns:
            output: Denoised trajectory (C, T, X, Y).
        """

        x_t = rearrange(
            self._create_blankets(x_t),
            "B C K X Y -> B (C K X Y)",
        )

        sigma_t = sigma_t * torch.ones(
            x_t.shape[0],
            1,
        ).to(x_t.device)

        x_t = self.denoiser(x_t, sigma_t, cond)

        return self._extract_states(
            rearrange(
                x_t,
                "B (C K X Y) -> B C K X Y",
                C=self.C,
                K=self.blanket_size,
                X=self.X,
                Y=self.Y,
            )
        )

    def _create_blankets(self, x_t: Tensor) -> Tensor:
        r"""Transform trajectory into overlapping blankets."""
        self.trajectory_size = x_t.shape[1]
        x_t = x_t.unfold(dimension=1, size=self.blanket_size, step=1)
        x_t = rearrange(x_t, "C B X Y K -> B C K X Y")
        return x_t

    def _extract_states(self, x_t: Tensor) -> Tensor:
        r"""Transform overlapping blankets into trajectory."""

        B, _, K, _, _ = x_t.shape

        if B == 1:
            return x_t[0]

        elif B == 2:
            return torch.concat(
                [x_t[0, :, : self.blanket_size], x_t[1, :, -(self.trajectory_size - K) :]], dim=1
            )

        else:
            idx_start = self.blanket_size - self.blanket_neighbors
            x_start = x_t[0, :, :idx_start]
            x_end = x_t[-1, :, -idx_start:]
            x_middle = torch.cat(
                [x_t[i, :, self.blanket_size // 2].unsqueeze(1) for i in range(1, B - 1)], dim=1
            )

            return torch.cat([x_start, x_middle, x_end], dim=1)
