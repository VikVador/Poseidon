r"""Optimizers."""

import torch
import torch.nn as nn

from torch import Tensor
from torch.optim import AdamW, Optimizer
from typing import Dict, Iterable, Optional

# isort: split
from poseidon.training.soap import SOAP


def get_optimizer(
    nn_parameters: Iterable[nn.Parameter],
    config_optimizer: Dict,
) -> Optimizer:
    r"""Initialize an optimizer based on the provided configuration.

    Arguments:
        nn_parameters: Iterable containing the parameters to optimize.
        config_optimizer: Dictionary containing optimizer settings.
    """

    optimizer_type = config_optimizer.get("optimizer")

    if optimizer_type == "adamw":
        optimizer = AdamW(
            nn_parameters,
            lr=config_optimizer.get("learning_rate"),
            weight_decay=config_optimizer.get("weight_decay"),
            betas=config_optimizer.get("betas")[:2],
        )

    elif optimizer_type == "soap":
        optimizer = SOAP(
            nn_parameters,
            lr=config_optimizer.get("learning_rate"),
            weight_decay=config_optimizer.get("weight_decay"),
            betas=config_optimizer.get("betas"),
        )
    else:
        raise NotImplementedError(f"ERROR - Optimizer {optimizer_type} is not supported.")

    return optimizer


def safe_gd_step(
    optimizer: torch.optim.Optimizer,
    grad_clip: Optional[float] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Tensor:
    r"""Applies a gradient descent (GD) optimization step.

    To prevent invalid parameters, steps are skipped if not-a-number (NaN) or infinite
    values are found in the gradient. This feature requires CPU-GPU synchronization,
    which could be a bottleneck for some applications.

    Arguments:
        optimizer: An optimizer.
        grad_clip: The maximum gradient norm. If :py:`None`, gradients are not clipped.
        scaler: A gradient scaler for AMP training. It is considered already applied.

    Returns:
        The unclipped gradient norm.
    """

    if scaler:
        scaler.unscale_(optimizer)

    params = [p for group in optimizer.param_groups for p in group["params"]]

    if grad_clip is None:
        norm = torch.linalg.vector_norm(
            torch.stack([
                torch.linalg.vector_norm(p.grad) for p in params if torch.is_tensor(p.grad)
            ])
        )
    else:
        norm = nn.utils.clip_grad_norm_(params, grad_clip)

    if scaler:
        scaler.step(optimizer)
        scaler.update()
    elif norm.isfinite():
        optimizer.step()

    # Reseting gradients
    optimizer.zero_grad()
