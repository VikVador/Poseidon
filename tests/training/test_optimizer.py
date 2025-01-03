r"""Tests for the poseidon.training.optimizer module."""

import pytest
import torch.nn as nn

from torch.optim import AdamW

# isort: split
from poseidon.training.optimizer import get_optimizer
from poseidon.training.soap import SOAP


@pytest.fixture
def fake_network_parameters():
    """Fixture to create mock neural network parameters."""

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)

    return FakeModel().parameters()


def test_get_optimizer_adamw(fake_network_parameters):
    """Testing that the AdamW optimizer is initialized correctly."""
    config = {
        "optimizer": "adamw",
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "betas": (0.9, 0.999),
    }
    optimizer = get_optimizer(fake_network_parameters, config)
    assert isinstance(optimizer, AdamW), "ERROR - Expected an AdamW optimizer."
    assert (
        optimizer.defaults["lr"] == config["learning_rate"]
    ), f"ERROR - Expected learning rate to be 0.001 but got {optimizer.defaults['lr']}."
    assert (
        optimizer.defaults["weight_decay"] == config["weight_decay"]
    ), f"ERROR - Expected weight decay to be 0.01 but got {optimizer.defaults['weight_decay']}."


def test_get_optimizer_soap(fake_network_parameters):
    """Testing that the SOAP optimizer is initialized correctly."""
    config = {
        "optimizer": "soap",
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "betas": (0.9, 0.999, 0.999),
    }
    optimizer = get_optimizer(fake_network_parameters, config)
    assert isinstance(optimizer, SOAP), "ERROR - Expected a SOAP optimizer."
    assert (
        optimizer.defaults["lr"] == config["learning_rate"]
    ), f"ERROR - Expected learning rate to be 0.001 but got {optimizer.defaults['lr']}."
    assert (
        optimizer.defaults["weight_decay"] == config["weight_decay"]
    ), f"ERROR - Expected weight decay to be 0.01 but got {optimizer.defaults['weight_decay']}."


def test_get_optimizer_unsupported(fake_network_parameters):
    """Testing that an unsupported optimizer raises an exception."""
    config = {"optimizer": "random_optimizer"}
    with pytest.raises(NotImplementedError, match="ERROR - Optimizer"):
        get_optimizer(fake_network_parameters, config)
