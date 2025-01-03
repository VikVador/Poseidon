r"""Schedulers."""

import math

from torch.optim import Optimizer, lr_scheduler
from typing import Dict


def get_scheduler(
    optimizer: Optimizer,
    total_steps: int,
    config_scheduler: Dict,
) -> lr_scheduler:
    r"""Initialize a learning rate scheduler based on the provided configuration.

    Arguments:
        optimizer: Optimizer to be scheduled.
        total_steps: Total number of training steps.
        config_scheduler: Dictionary containing scheduler configuration.
    """

    scheduler_type = config_scheduler.get("scheduler")

    if scheduler_type == "constant":
        lr_lambda = lambda t: 1
    elif scheduler_type == "linear":
        lr_lambda = lambda t: max(0, 1 - (t / total_steps))
    elif scheduler_type == "cosine":
        lr_lambda = lambda t: (1 + math.cos(math.pi * t / total_steps)) / 2
    elif scheduler_type == "exponential":
        lr_lambda = lambda t: math.exp(math.log(1e-6) * t / total_steps)
    else:
        raise NotImplementedError(f"ERROR - Scheduler '{scheduler_type}' is not supported.")

    return lr_scheduler.LambdaLR(optimizer, lr_lambda)
