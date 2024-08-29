r"""Diffusion helpers"""

import torch
import torch.nn as nn

from torch import Tensor
from typing import Any, Callable


class Denoiser(nn.Module):
    r"""Denoiser model with EDM-style preconditioning.

    .. math:: d(x_t) \approx E[x | x_t]

    References:
        | Elucidating the Design Space of Diffusion-Based Generative Models (Karras et al., 2022)
        | https://arxiv.org/abs/2206.00364

    Arguments:
        backbone: A noise conditional network.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()

        self.backbone = backbone

    def forward(self, xt: Tensor, sigma_t: Tensor, context: Any = None) -> Tensor:
        r"""
        Arguments:
            xt: The noisy tensor, with shape :math:`(*, D)`.
            sigma_t: The noise std, with shape :math:`(*)`.

        Returns:
            The denoised tensor :math:`d(x_t)`, with shape :math:`(*, D)`.
        """

        c_skip = 1 / (sigma_t**2 + 1)
        c_out = sigma_t / torch.sqrt(sigma_t**2 + 1)
        c_in = 1 / torch.sqrt(sigma_t**2 + 1)
        c_noise = torch.log(sigma_t)

        c_skip, c_out, c_in = c_skip[..., None], c_out[..., None], c_in[..., None]

        if context is None:
            x_out = self.backbone(c_in * xt, c_noise)
        else:
            x_out = self.backbone(c_in * xt, c_noise, context)

        return c_skip * xt + c_out * x_out


class DenoiserLoss(nn.Module):
    r"""Loss for a denoiser model.

    .. math:: \lambda_t || d(x_t) - x ||^2

    Arguments:
        denoiser: A denoiser model :math:`d(x_t) \approx E[x | x_t]`.
    """

    def __init__(self, denoiser: Denoiser):
        super().__init__()

        self.denoiser = denoiser

    def forward(self, x: Tensor, sigma_t: Tensor, context: Any = None) -> Tensor:
        r"""
        Arguments:
            x: The clean tensor, with shape :math:`(*, D)`.
            sigma_t: The noise std, with shape :math:`(*)`.
            context: An optional context passed to the denoiser.

        Returns:
            The reduced denoising loss, with shape :math:`()`.
        """

        lmb_t = 1 / sigma_t**2 + 1

        z = torch.randn_like(x)
        xt = x + sigma_t[..., None] * z

        dxt = self.denoiser(xt, sigma_t, context)

        error = dxt - x

        return torch.mean(lmb_t * torch.mean(error**2, dim=-1))


class NoiseSchedule(nn.Module):
    r"""Log-linear noise schedule.

    .. math:: \sigma_t = \exp(\log(a) (1 - t) + \log(b) t)

    Arguments:
        a: The noise lower bound.
        b: The noise upper bound.
    """

    def __init__(self, a: float = 1e-3, b: float = 1e2):
        super().__init__()

        self.register_buffer("a", torch.as_tensor(a).log())
        self.register_buffer("b", torch.as_tensor(b).log())

    def forward(self, t: Tensor) -> Tensor:
        r"""
        Arguments:
            t: The schedule time, with shape :math:`(*)`.

        Returns:
            The noise std :math:`\sigma_t`, with shape :math:`(*)`.
        """

        return torch.exp(self.a + (self.b - self.a) * t)


class DDPM(nn.Module):
    r"""DDPM sampler for the reverse SDE.

    .. math:: x_s = x_t - \tau (x_t - d(x_t)) + \sigma_s \sqrt{\tau} \epsilon

    where :math:`\tau = 1 - \frac{\sigma_s^2}{\sigma_t^2}`.

    Arguments:
        denoiser: A denoiser model :math:`d(x_t) \approx E[x | x_t]`.
        schedule: The noise schedule.
        steps: The number of sampling steps.
    """

    def __init__(
        self,
        denoiser: Callable[[Tensor, Tensor], Tensor],
        schedule: NoiseSchedule = None,
        steps: int = 256,
    ):
        super().__init__()

        self.denoiser = denoiser

        if schedule is None:
            self.schedule = NoiseSchedule()
        else:
            self.schedule = schedule

        self.steps = steps

    def forward(self, x1: Tensor) -> Tensor:
        r"""
        Arguments:
            x1: A noise tensor from :math:`p(x_1)`, with shape :math:`(*, D)`.

        Returns:
            A data tensor from :math:`p(x_0 | x_1)`, with shape :math:`(*, D)`.
        """

        dt = torch.as_tensor(1 / self.steps).to(x1)
        time = torch.linspace(1, dt, self.steps).to(x1)

        xt = x1
        for t in time:
            xt = self.step(xt, t, t - dt)

        return xt

    def step(self, xt: Tensor, t: Tensor, s: Tensor) -> Tensor:
        sigma_s, sigma_t = self.schedule(s), self.schedule(t)
        tau = 1 - (sigma_s / sigma_t) ** 2
        eps = torch.randn_like(xt)

        return xt - tau * (xt - self.denoiser(xt, sigma_t)) + sigma_s * torch.sqrt(tau) * eps


class DDIM(DDPM):
    r"""DDIM sampler for the reverse SDE.

    .. math:: x_s = x_t - (1 - \frac{\sigma_s}{\sigma_t}) (x_t - d(x_t))

    Arguments:
        denoiser: A denoiser model :math:`d(x_t) \approx E[x | x_t]`.
        schedule: The noise schedule.
    """

    def step(self, xt: Tensor, t: Tensor, s: Tensor) -> Tensor:
        sigma_s, sigma_t = self.schedule(s), self.schedule(t)

        return xt - (1 - sigma_s / sigma_t) * (xt - self.denoiser(xt, sigma_t))
