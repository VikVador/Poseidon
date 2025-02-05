r"""Diffusion denoiser model."""

import torch
import torch.nn as nn

from torch import Tensor

# isort: split
from poseidon.diffusion.backbone import PoseidonBackbone


class PoseidonDenoiser(nn.Module):
    r"""Denoiser model with EDM-style preconditioning for diffusion models.

    Information:
        Preconditioning terms (c_skip, c_out, etc.) are derived from the noise
        level sigma and help guide the denoising process by modulating the
        input tensor and the noise predictions.

    Formulation:
        D_theta(x, sigma, c) = c_skip * x + c_out * Backbone(c_in(sigma) * x, c_noise(sigma)).

    References:
        | Elucidating the Design Space of Diffusion-Based Generative Models (Karras et al., 2022).
        | https://arxiv.org/abs/2206.00364

    Arguments:
        backbone: A `PoseidonBackbone` instance.
    """

    def __init__(self, backbone: PoseidonBackbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x_t: Tensor, sigma_t: Tensor) -> Tensor:
        r"""Denoising using EDM-style preconditioning.

        Information:
            Since the dataset is standardized, sigma(data)^2 = 1.

        Arguments:
            x_t: Noisy input tensor (B, D).
            sigma_t: Noise level (B, 1).

        Returns:
            Tensor: Denoised tensor (B, D).
        """
        # fmt:off
        c_skip  = 1       / (sigma_t**2 + 1)            # Retains part of the original noisy signal
        c_out   = sigma_t / torch.sqrt(sigma_t**2 + 1)  # Scales the denoised output
        c_in    = 1       / torch.sqrt(sigma_t**2 + 1)  # Modulates the input tensor to account for noise level
        c_noise = 1e1     * torch.log(sigma_t)          # Logarithmic noise level used in conditioning, scaled by 1e1 to have a wider range for learning

        # Denoising (1)
        x_t_denoised = self.backbone(x = c_in * x_t, sigma = c_noise)

        # Denoising with skip connection (2)
        return c_skip * x_t + c_out * x_t_denoised
