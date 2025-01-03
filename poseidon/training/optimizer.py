r"""Optimizers."""

import torch.nn as nn

from torch.optim import AdamW, Optimizer
from typing import Dict, Iterable

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
