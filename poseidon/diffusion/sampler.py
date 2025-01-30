r"""Diffusion sampler."""

import torch
import torch.nn as nn

from abc import abstractmethod
from torch import Tensor
from tqdm import tqdm

# isort: split
from poseidon.diffusion.denoiser import PoseidonDenoiser
from poseidon.score.prior import PoseidonScorePrior

# Constants
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class PoseidonEDMSampler(nn.Module):
    r"""Template to create a diffusion sampler

    References:
        | Elucidating the Design Space of Diffusion-Based Generative Models (Karras et al., 2022).
        | https://arxiv.org/abs/2206.00364
    """

    def __init__(
        self,
        denoiser: PoseidonDenoiser,
        parallelize: bool = False,
        rho: float = 2,
        sigma_min: float = 0.002,
        sigma_max: float = 80,
    ):
        super().__init__()

        self.score_prior = PoseidonScorePrior(
            denoiser,
            parallelize=parallelize,
        )

        # Extracting dimensions of the region
        self.C, self.H, self.W = (
            denoiser.backbone.C,
            denoiser.backbone.H,
            denoiser.backbone.W,
        )

        # Properties for generating timesteps
        self.steps, self.rho, self.sigma_min_rho_, self.sigma_max_rho_ = (
            1,
            rho,
            sigma_min ** (1 / rho),
            sigma_max ** (1 / rho),
        )

    def get_timestep(self, i: int) -> Tensor:
        r"""Computes timestep at a given step of the diffusion process."""
        return (
            self.sigma_max_rho_
            + (i / (self.steps - 1)) * (self.sigma_min_rho_ - self.sigma_max_rho_)
        ) ** self.rho

    def get_noise(self, t: int) -> Tensor:
        r"""Computes noise at a given timestep of the diffusion process."""
        return torch.tensor([t]).to(DEVICE)

    def get_noise_derivative(self, t: int) -> Tensor:
        r"""Computes noise derivative at a given timestep of the diffusion process."""
        return torch.tensor([1]).to(DEVICE)

    def evaluate(self, x: Tensor, t: int) -> Tensor:
        r"""Evaluates the function f(x_i, t_i)."""
        return (
            self.get_noise_derivative(t)
            * self.get_noise(t)
            * self.score_prior.forward(x, self.get_noise(t).unsqueeze(0))
        )

    @abstractmethod
    def update(self, x_i: Tensor, t_i: int, h_i: float) -> Tensor:
        r"""Perfoms one-step of diffusion."""
        raise NotImplementedError

    def forward(self, trajectory_size: int, forecast_size: int = 1, steps: int = 64) -> Tensor:
        r"""Generates forecasts of a trajectory."""
        with torch.no_grad():
            progression = tqdm(
                total=steps,
                desc="| POSEIDON | Diffusion",
            )

            forecasts, self.steps = [], steps

            for _ in range(forecast_size):
                x_i = self.get_noise(t=self.get_timestep(0)) * torch.randn(
                    self.C, trajectory_size, self.H, self.W
                ).to(DEVICE)

                for s in range(0, steps - 1):
                    t_i, h_i = (
                        self.get_timestep(s),
                        self.get_timestep(s + 1) - self.get_timestep(s),
                    )

                    x_i = self.update(
                        x_i=x_i,
                        t_i=t_i,
                        h_i=h_i,
                    )

                    if s % forecast_size == 0:
                        progression.update(1)

                forecasts.append(x_i)

            return torch.stack(forecasts, dim=0)


class PoseidonEulerSampler(PoseidonEDMSampler):
    r"""Euler 1st Order."""

    def update(self, x_i: Tensor, t_i: int, h_i: float) -> Tensor:
        r"""Perfoms one-step of diffusion."""
        return x_i - h_i * self.evaluate(x_i, t_i)


class PoseidonHeunSampler(PoseidonEDMSampler):
    r"""Heun 2nd Order."""

    def update(self, x_i: Tensor, t_i: int, h_i: float) -> Tensor:
        r"""Perfoms one-step of diffusion."""
        evaluation = self.evaluate(x_i, t_i)
        x = x_i - h_i * evaluation
        return x_i - (h_i / 2) * (evaluation + self.evaluate(x=x, t=(t_i + h_i)))
