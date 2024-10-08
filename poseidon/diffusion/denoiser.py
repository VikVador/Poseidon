r"""Diffusion - Custom denoiser for diffusion pipeline."""

import torch
import torch.nn as nn

# isort: split
from poseidon.diffusion.backbone import PoseidonBackbone


class PoseidonDenoiser(nn.Module):
    r"""Denoiser model with EDM-style preconditioning for diffusion models.

    The preconditioning terms (c_skip, c_out, etc.) are derived from the noise
    level sigma and help guide the denoising process by modulating the input tensor
    and the noise predictions.

    Formulation:
        D_theta(x, sigma, c) = c_skip * x + c_out * F_theta(c_in(sigma) * x, c_noise(sigma)).

    References:
        | Elucidating the Design Space of Diffusion-Based Generative Models (Karras et al., 2022).
        | https://arxiv.org/abs/2206.00364
    """

    def __init__(self, backbone: PoseidonBackbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x_t: torch.Tensor, sigma_t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        r"""Forward pass of the denoiser model using EDM-style preconditioning.

        Arguments:
            x_t: Noisy input tensor with shape (B, D).
            sigma_t: Noise level (or diffusion step) with shape (B, 1).
            c: Tokenized time-based conditioning (B, 3), where the three elements correspond to month, day, hour.

        Returns:
            The denoised tensor (B, D).
        """
        # fmt:off
        # Note: Since the dataset is standardized, sigma(data)^2 = 1
        # Preconditioning terms based on the noise level sigma
        c_in    = 1       / torch.sqrt(sigma_t**2 + 1)  # Modulates the input tensor to account for noise level
        c_out   = sigma_t / torch.sqrt(sigma_t**2 + 1)  # Scales the denoised output
        c_skip  = 1       / (sigma_t**2 + 1)            # Retains part of the original noisy signal
        c_noise = torch.log(sigma_t).squeeze(1)         # Logarithmic noise level used in conditioning

        # Denoising a blanket
        x_t_denoised = self.backbone(x = c_in * x_t, sigma = c_noise, c = c)

        # EDM-style preconditioning
        return c_skip * x_t + c_out * x_t_denoised
