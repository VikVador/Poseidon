r"""Tests for UNet architecture in poseidon.network.unet module."""

import pytest
import torch

from pathlib import Path
from typing import Optional

# isort: split
from poseidon.network.unet import UNet


@pytest.mark.parametrize("in_channels, out_channels", [(1, 1), (1, 2)])
@pytest.mark.parametrize("time", [3, 5])
@pytest.mark.parametrize("latitude", [16])
@pytest.mark.parametrize("longitude", [16])
@pytest.mark.parametrize("mod_features", [4])
@pytest.mark.parametrize("dropout", [None, 0.1])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("hid_channels", [(2, 4)])
@pytest.mark.parametrize("hid_blocks", [(1, 1)])
def test_unet(
    tmp_path: Path,
    in_channels: int,
    out_channels: int,
    time: int,
    latitude: int,
    longitude: int,
    mod_features: int,
    dropout: Optional[float],
    batch_size: int,
    hid_channels: tuple,
    hid_blocks: tuple,
):
    """Testing UNet model to verify tensors shape, gradient flow, as well as model save/load functionality."""
    make = lambda: UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        mod_features=mod_features,
        kernel_size=3,
        blanket_size=time,
        stride=2,
        hid_channels=hid_channels,
        hid_blocks=hid_blocks,
        dropout=dropout,
    )

    unet = make()
    unet.train()

    # Forward pass
    x = torch.randn(batch_size, in_channels, time, latitude, longitude)
    mod = torch.randn(batch_size, mod_features)
    y = unet(x, mod)

    # Shape
    assert y.ndim == x.ndim
    assert y.shape[0] == batch_size
    assert y.shape[1] == out_channels
    assert y.shape[2:] == x.shape[2:]

    # Gradients
    loss = y.square().sum()
    loss.backward()

    assert y.requires_grad, "ERROR - Output tensor should require gradients."
    for name, param in unet.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"ERROR - Gradient not computed for {name} : {param}"
        else:
            assert (
                param.grad is None
            ), f"ERROR - Unexpected gradient for non-learnable parameter: {name}"

    # Saving
    torch.save(unet.state_dict(), tmp_path / "state.pth")

    copy = make()
    copy.load_state_dict(torch.load(tmp_path / "state.pth", weights_only=True))

    unet.eval()
    copy.eval()

    y_unet = unet(x, mod)
    y_copy = copy(x, mod)

    assert torch.allclose(
        y_unet, y_copy, atol=1e-5
    ), "ERROR - Model outputs do not match after loading."
