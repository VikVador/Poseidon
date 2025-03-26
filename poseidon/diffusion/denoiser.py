r"""Diffusion denoisers."""

import torch
import torch.nn as nn

from functools import partial
from torch import Tensor
from typing import Callable

# isort: split
from poseidon.diffusion.backbone import PoseidonBackbone
from poseidon.diffusion.tools import PoseidonTrajectoryWrapper
from poseidon.math import gmres


class PoseidonDenoiser(nn.Module):
    r"""Denoiser model with EDM-style preconditioning for diffusion models.

    Formulation:
        D_theta(x_t, sigma_t) = c_skip * x_t + c_out * Backbone(c_in(sigma_t) * x_t, c_noise(sigma_t)).

    References:
        | Elucidating the Design Space of Diffusion-Based Generative Models (Karras et al., 2022).
        | https://arxiv.org/abs/2206.00364

    Arguments:
        backbone: A :class:`PoseidonBackbone` instance.
    """

    def __init__(self, backbone: PoseidonBackbone):
        super().__init__()
        self.backbone = backbone

    def forward(
        self,
        x_t: Tensor,
        sigma_t: Tensor,
    ) -> Tensor:
        r"""Denoising using EDM-style preconditioning.

        Arguments:
            x_t: Noisy input tensor (B, C * K * X * Y).
            sigma_t: Associated noise levels (B, 1).

        Returns:
            Cleaned tensor (B, C * K * X * Y).
        """
        # fmt:off
        c_skip  = 1       / (sigma_t**2 + 1)            # Retains part of the original noisy signal
        c_out   = sigma_t / torch.sqrt(sigma_t**2 + 1)  # Scales the denoised output
        c_in    = 1       / torch.sqrt(sigma_t**2 + 1)  # Modulates the input tensor to account for noise level
        c_noise = 1e1     * torch.log(sigma_t)          # Rescaling noise levels

        # Estimating (scaled) denoised state
        return c_skip * x_t + c_out * self.backbone(x_t = c_in * x_t, sigma_t = c_noise)


class PoseidonMMPSDenoiser(nn.Module):
    r"""Denoiser model with MMPS-style observation conditioning.

    References:
        | Learning Diffusion Priors from Observations by Expectation Maximization (Rozet et al., 2024)
        | https://arxiv.org/abs/2405.13712

    Information:
        Be careful to wrap the denoiser model with :class:`PoseidonTrajectoryWrapper`.

    Arguments:
        denoiser: A Gaussian denoiser.
        y: An observation y ~ 𝒩(Ax, Σᵧ) of shape (M).
        A: Observation operator x ↦ Ax. It should take in a vector x of shape (B, D) and return a vector of shape (B, M).
        cov_y: Covariance matrix or the noise variance Σᵧ if the covariance is diagonal, with shape (), (D), or (D, D).
        tweedie_covariance: Whether to use the Tweedie covariance formula or not. If False, use Σₜ instead.
        iterations: Number of solver iterations.
    """

    def __init__(
        self,
        denoiser: PoseidonTrajectoryWrapper,
        y: Tensor,
        A: Callable[[Tensor], Tensor],
        cov_y: Tensor,
        tweedie_covariance: bool = True,
        iterations: int = 1,
    ):
        super().__init__()

        self.A = A
        self.denoiser = denoiser
        self.tweedie_covariance = tweedie_covariance
        self.register_buffer("y", torch.as_tensor(y))
        self.register_buffer("cov_y", torch.as_tensor(cov_y))

        self.solve = partial(gmres, iterations=iterations)

    def forward(self, x_t: Tensor, sigma_t: Tensor, **kwargs):
        r"""Denoising with MMPS-style observation conditioning."""

        cov_t = sigma_t**2

        with torch.enable_grad():
            x_t = x_t.detach().requires_grad_()
            x_hat = self.denoiser(x_t, sigma_t, **kwargs)
            y_hat = self.A(x_hat)

        def A_lin(v):
            return torch.func.jvp(self.A, (x_hat,), (v,))[-1]

        def At(v):
            return torch.autograd.grad(y_hat, x_hat, v, retain_graph=True)[0]

        if self.tweedie_covariance:
            if len(self.cov_y.shape) <= 1:
                cov_y = lambda v: self.cov_y * v + A_lin(
                    cov_t * torch.autograd.grad(x_hat, x_t, At(v), retain_graph=True)[0]
                )
            else:
                # Matrix - batched vector product: (D, D) @ (B, D) -> (B, D)
                cov_y = lambda v: torch.einsum("ij, bj->bi", self.cov_y, v) + A_lin(
                    cov_t * torch.autograd.grad(x_hat, x_t, At(v), retain_graph=True)[0]
                )
        else:
            cov_y = lambda v: cov_t * v

        grad = self.y - y_hat
        grad = self.solve(A=cov_y, b=grad)
        score = torch.autograd.grad(y_hat, x_t, grad)[0]

        return x_hat + cov_t * score
