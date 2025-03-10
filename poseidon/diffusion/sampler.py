r"""Diffusion samplers.

From:
  | azula library (François Rozet)
  | https://github.com/francois-rozet/azula

"""

import torch
import torch.nn as nn

from abc import abstractmethod
from torch import Tensor
from tqdm import tqdm
from typing import Tuple

# isort: split
from poseidon.diffusion.denoiser import PoseidonDenoiser
from poseidon.diffusion.schedulers import PoseidonNoiseScheduler
from poseidon.math import gauss_legendre

# fmt: off
#
# Constants
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class Sampler(nn.Module):
    r"""Template to create a diffusion sampler

    Mathematics:

            xₛ = xₜ - τ (xₜ - d(xₜ)) + σₛ √τ ε

    where τ is determined by the noise schedule.

    Arguments:
        denoiser: A denoiser model d(xₜ) ≈ E[x | xₜ]
        schedule: A noise schedule.
        dimensions: Dimensions of the trajectory (C, X, Y).
    """

    def __init__(
        self,
        denoiser: PoseidonDenoiser,
        schedule: nn.Module,
        dimensions: Tuple[int, int, int],
        *args,
        **kwargs,
    ):
        super().__init__()

        self.C, self.X, self.Y = dimensions

        self.denoiser = denoiser
        self.schedule = schedule
        self.steps    = 1

    @abstractmethod
    def forward(self, x1: Tensor) -> Tensor:
        r"""
        Arguments:
            x1: A noise tensor from p(x₁), with shape (*, D).

        Returns:
            A data tensor from p(x₀ | x₁), with shape (*, D).
        """
        pass


class LMSSampler(Sampler):
    r"""Creates a linear multi-step (LMS) sampler.

    References:
        | k-diffusion (Katherine Crowson)
        | https://github.com/crowsonkb/k-diffusion

    Arguments:
        denoiser: A denoiser model d(xₜ) ≈ E[x | xₜ].
        schedule: A noise schedule.
        order: Order of the multi-step method.
        kwargs: Keyword arguments passed to Sampler.
    """

    def __init__(
        self,
        denoiser: PoseidonDenoiser,
        schedule: PoseidonNoiseScheduler,
        dimensions: Tuple[int, int, int],
        order: int = 3,
    ):
        super().__init__(denoiser=denoiser, schedule=schedule, dimensions=dimensions)
        self.order = order

    @staticmethod
    def adams_bashforth(t: Tensor, i: int, order: int = 3) -> Tensor:
        r"""Returns the coefficients of the N-th order Adams-Bashforth method.

        Wikipedia:
            https://wikipedia.org/wiki/Linear_multistep_method

        Arguments:
            t: Integration variable, with shape (T).
            i: Integration step.
            order: Method order N.

        Returns:
            Adams-Bashforth coefficients, with shape (N).
        """

        ti = t[i]
        tj = t[i - order : i]
        tk = torch.cat((tj, tj)).unfold(0, order, 1)[:order, 1:]
        tj_tk = tj[..., None] - tk

        # Lagrange basis
        def lj(t):
            return torch.prod((t[..., None, None] - tk) / tj_tk, dim=-1)

        # Adams-Bashforth coefficients
        return gauss_legendre(lj, tj[-1], ti, n=order // 2 + 1)

    @torch.no_grad()
    def forward(
        self,
        trajectory_size: int,
        forecast_size: int,
        steps: int = 32,
    ) -> Tensor:
        r"""Generating forecasts."""

        forecasts, self.steps, progression = (
            [],
            steps,
            tqdm(
                total=steps,
                desc="| POSEIDON | Diffusion",
            ),
        )

        for _ in range(forecast_size):

            # Initial Noise
            sigma_t = self.schedule(torch.tensor([1]))

            # Initial state (random Gaussian noise)
            xt = (sigma_t * torch.randn(self.C, trajectory_size, self.X, self.Y)).to(DEVICE)

            # Other initializations
            time   = torch.linspace(1, 0, self.steps + 1).to(DEVICE)
            sigmas = self.schedule(time).squeeze()
            ratio  = sigmas.double()

            # Stores N past derivatives for Adams-Bashforth
            derivatives = []

            # Solving reverse-time diffusion
            for s, sigma_t in enumerate(sigmas[:-1]):

                # Estimating reconstructed state
                q_t = self.denoiser(xt, sigma_t)

                # Computing noise to remove
                z_t = (xt - q_t) / sigma_t

                # Storing derivatives
                derivatives.append(z_t)
                if len(derivatives) > self.order:
                    derivatives.pop(0)

                # Adams-Bashforth coefficients
                coefficients = self.adams_bashforth(ratio, s + 1, order=len(derivatives))
                coefficients = coefficients.to(xt)
                delta        = sum(c * d for c, d in zip(coefficients, derivatives))

                # Updating state
                xt = xt + delta

                # Progression bar
                if s % forecast_size == 0:
                    progression.update(1)

            # Storing forecast
            forecasts.append(xt)

        return torch.stack(forecasts, dim=0)
