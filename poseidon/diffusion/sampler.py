r"""Diffusion samplers."""

import torch
import torch.nn as nn

from abc import abstractmethod
from torch import Tensor
from tqdm import tqdm

# isort: split
from poseidon.diffusion.denoiser import PoseidonDenoiser
from poseidon.score.prior import PoseidonScorePrior

# fmt: off
#
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
        rho: float = 2,
        sigma_min: float = 0.01,
        sigma_max: float = 80,
    ):
        super().__init__()

        # Computes unconditional score
        self.score_prior = PoseidonScorePrior(denoiser)

        # Extracting dimensions of the region
        self.C, self.H, self.W = (
            denoiser.backbone.C,
            denoiser.backbone.X,
            denoiser.backbone.Y,
        )

        # Properties of timesteps
        self.steps, self.rho, self.sigma_min_rho_, self.sigma_max_rho_ = (
            1,
            rho,
            sigma_min ** (1 / rho),
            sigma_max ** (1 / rho),
        )

    def get_timestep(self, i: int) -> Tensor:
        r"""Computes timestep at a given step of the reverse diffusion process."""
        return (
            self.sigma_max_rho_
            + (i / (self.steps - 1)) * (self.sigma_min_rho_ - self.sigma_max_rho_)
        ) ** self.rho

    def get_noise(self, t: int) -> Tensor:
        r"""Computes noise."""
        return torch.tensor([t]).to(DEVICE)

    def get_noise_derivative(self, t: int) -> Tensor:
        r"""Computes noise derivative."""
        return torch.tensor([1]).to(DEVICE)

    def evaluate(self, x: Tensor, t: int) -> Tensor:
        r"""Evaluates the function f(x_i, t_i)."""
        return (
            -self.get_noise_derivative(t)
            * self.get_noise(t)
            * self.score_prior.forward(x, self.get_noise(t).unsqueeze(0))
        )

    @abstractmethod
    def step(self, x_i: Tensor, t_i: int, h_i: float) -> Tensor:
        r"""Computes one-step of diffusion."""
        raise NotImplementedError

    def forward(
        self,
        trajectory_size: int,
        forecast_size: int = 3,
        steps: int = 64,
    ) -> Tensor:
        r"""Generating forecasts."""
        with torch.no_grad():

            forecasts, self.steps, progression = (
                [],
                steps,
                tqdm(
                    total=steps,
                    desc="| POSEIDON | Diffusion",
                ),
            )

            for _ in range(forecast_size):

                # Initial state (random Gaussian noise)
                x_i = self.get_noise(
                    t=self.get_timestep(0),
                ) * torch.randn(self.C, trajectory_size, self.H, self.W).to(DEVICE)

                for s in range(0, steps - 1):
                    x_i = self.step(
                        x_i=x_i,
                        t_i=self.get_timestep(s),
                        h_i=self.get_timestep(s + 1) - self.get_timestep(s),
                    )

                    if s % forecast_size == 0:
                        progression.update(1)

                forecasts.append(x_i)

            return torch.stack(forecasts, dim=0)


class PoseidonEulerSampler(PoseidonEDMSampler):
    r"""A numerical solver using Euler (1st Order) solver."""

    def step(self, x_i: Tensor, t_i: int, h_i: float) -> Tensor:
        r"""Perfoms one-step of diffusion."""
        return x_i + h_i * self.evaluate(x_i, t_i)


class PoseidonHeunSampler(PoseidonEDMSampler):
    r"""A numerical solver using Heun (2nd Order) solver."""

    def step(self, x_i: Tensor, t_i: int, h_i: float) -> Tensor:
        r"""Perfoms one-step of diffusion."""

        # Storing the evaluation to only call twice the model
        f_i = self.evaluate(x_i, t_i)

        # Euler Step
        x = x_i + h_i * f_i

        # Correction (Heun)
        return x_i + (h_i / 2) * (f_i + self.evaluate(x=x, t=(t_i + h_i)))
