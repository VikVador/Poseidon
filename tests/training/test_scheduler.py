r"""Tests for the poseidon.training.optimizer module."""

import pytest
import torch

from torch.optim import SGD

# isort: split
from poseidon.training.scheduler import get_scheduler


@pytest.fixture
def fake_optimizer():
    """Fixture to create a dummy optimizer."""
    params = [torch.zeros(1, requires_grad=True)]
    return SGD(params, lr=0.1)


def test_scheduler_unsupported(fake_optimizer):
    """Testing that unsupported scheduler types raise an exception."""
    config = {"scheduler": "unsupported"}
    total_steps = 100
    with pytest.raises(NotImplementedError, match="ERROR - Scheduler"):
        get_scheduler(fake_optimizer, total_steps, config)
