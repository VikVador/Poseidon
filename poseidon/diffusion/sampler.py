r"""Diffusion - Custom sampler to perform diffusion."""

import torch
import torch.nn as nn

from einops import rearrange
from tqdm import trange
from typing import Tuple

# isort: split
from datetime import datetime, timedelta
from poseidon.diffusion.denoiser import PoseidonDenoiser
from poseidon.diffusion.tools import (
    compute_blanket_indices,
    extract_blankets_in_trajectories,
    time_tokenizer,
)


class PoseidonSampler(nn.Module):
    r"""A sampler for a diffusion model, responsible for generating trajectories from a denoiser model.

    Arguments:
        denoiser: The denoiser model to be used for the diffusion process.
        steps: Number of diffusion steps to perform.
        sigma_min: Minimum noise value for diffusion.
        sigma_max: Maximum noise value for diffusion.
        rho: Smoothing factor that controls the curvature of the noise schedule.
    """

    def __init__(
        self,
        denoiser: PoseidonDenoiser,
        steps: int = 256,
        sigma_min: float = 0.002,
        sigma_max: float = 80,
        rho: int = 6,
    ):
        super().__init__()

        self.denoiser = denoiser.cuda()
        self.k = denoiser.backbone.k
        self.blanket_size = denoiser.backbone.blanket_size
        self.channels, self.latitude, self.longitude = (
            denoiser.backbone.channels,
            denoiser.backbone.latitude,
            denoiser.backbone.longitude,
        )

        # Precomputing timesteps
        sigma_min, sigma_max, rho = 0.002, 80, 6
        steps_tensor = torch.arange(steps + 1)
        sigma_max_rho_ = sigma_max ** (1 / rho)
        sigma_min_rho_ = sigma_min ** (1 / rho)
        self.timesteps = (
            sigma_max_rho_ + (steps_tensor / (steps)) * (sigma_min_rho_ - sigma_max_rho_)
        ) ** rho

    def score_prior(
        self,
        x_i: torch.Tensor,
        noise_i: torch.Tensor,
        time: torch.Tensor,
        trajectory_size: int,
        idx=Tuple[int, int, int],
    ) -> torch.Tensor:
        """Computes the score for the prior.

        Arguments:
            x_i: The current state
            noise_i: The noise
            time: The time token
            trajectory_size: The number of days in the trajectory
            idx: The indices of the blankets
        """

        # Preprocessing as batch of blankets
        x_i = x_i.repeat(trajectory_size, 1, 1, 1, 1)
        x_i = extract_blankets_in_trajectories(x=x_i, blanket_idx=(idx[0], idx[1]))
        x_i = rearrange(
            x_i,
            "N ... -> N (...)",
        )

        # Scores (for each blanket)
        score_blankets = (
            self.denoiser(x_i.cuda(), noise_i.cuda(), time.cuda()).cpu() - x_i
        ) / noise_i**2

        score_blankets = rearrange(
            score_blankets,
            "N (C K H W) -> N C K H W",
            K=self.blanket_size,
            C=self.channels,
            H=self.latitude,
            W=self.longitude,
        )
        s = torch.stack([score_blankets[i, :, idx] for i, idx in enumerate(idx[2])])
        s = rearrange(s, "K C H W -> C K H W")

        # Score (individual states)
        return s

    def _initialize_state(self, trajectory_size: int) -> torch.Tensor:
        """Initializes the state for the diffusion process."""
        return (
            torch.randn(self.channels, trajectory_size, self.latitude, self.longitude)
            * self.timesteps[0]
        )

    def _initialize_time(self, trajectory_size: int, date: str) -> torch.Tensor:
        """Converts a date string into a tokenized time tensor."""
        return time_tokenizer(
            torch.tensor(
                [
                    [d.month + 1, d.day, d.hour]
                    for t in range(trajectory_size)
                    for d in [datetime.strptime(f"{date}-12", "%Y-%m-%d-%H") + timedelta(days=t)]
                ],
                dtype=torch.int,
            )
        )

    def forward(self, trajectory_size: int, date: str) -> torch.Tensor:
        r"""Generates a forecast using the diffusion process.

        Arguments:
            trajectory_size (int): Number of steps (days) in the forecasted trajectory.
            date (str): The starting date of the forecast, in "YEAR-MM-DD" format.

        Returns:
            torch.Tensor: A generated forecast trajectory with dimensions (trajectory_size, channels, latitude, longitude).
        """
        # fmt: off
        # Pre-computing blanket indices
        idx_start, idx_end, idx_state = compute_blanket_indices(indices=torch.arange(trajectory_size), k=self.k, trajectory_size=trajectory_size)

        with torch.no_grad():

            # Initiliazation of the state and time
            x_i, time = self._initialize_state(trajectory_size), self._initialize_time(trajectory_size, date)

            # --- Diffusion ---
            for t in trange(len(self.timesteps) - 1):

                # Constants
                t_i, delta_t, noise_i = (
                        self.timesteps[t],
                        self.timesteps[t + 1] - self.timesteps[t],
                        torch.ones((trajectory_size, 1)) * self.timesteps[t]
                    )

                # Computing score (prior and observation)
                score = self.score_prior(x_i, noise_i, time, trajectory_size, [idx_start, idx_end, idx_state])

                # Integration step (Euler 1st order)
                x_i = x_i - t_i * delta_t * score

        return x_i
