r"""Tests for the poseidon.data.mappings module."""

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from pathlib import Path

# isort: split
from poseidon.data.mappings import (
    from_tensor_to_indices,
    from_tensor_to_xarray,
)

FAKE_REGION = {
    "latitude": slice(0, 2),
    "longitude": slice(0, 2),
    "level": slice(0, 4),
}


# fmt: off
@pytest.fixture
def fake_black_sea_dataset(tmp_path) -> Path:
    """Fixture to provide a fake black sea dataset."""
    time = pd.date_range("1995-01-01", periods=12, freq="d")
    latitude = np.array([0, 1, 2, 3], dtype=np.float32)
    longitude = np.array([0, 1, 2, 3], dtype=np.float32)
    level = np.array([0, 1, 2, 2], dtype=np.int64)

    structure = np.ones((len(time), len(level), len(latitude), len(longitude)), dtype=np.float32)
    chl = 1 * structure
    dox = 2 * structure
    rho = 3 * structure
    sal = 4 * structure
    tmp = 5 * structure
    ssh = 6 * structure

    ds = xr.Dataset(
        {
            "CHL": (["time", "level", "latitude", "longitude"], chl),
            "DOX": (["time", "level", "latitude", "longitude"], dox),
            "rho": (["time", "level", "latitude", "longitude"], rho),
            "vosaline": (["time", "level", "latitude", "longitude"], sal),
            "votemper": (["time", "level", "latitude", "longitude"], tmp),
            "ssh": (["time", "latitude", "longitude"], ssh[:, 0]),
        },
        coords={
            "time": time,
            "level": level,
            "latitude": latitude,
            "longitude": longitude,
        },
    )

    path_data = tmp_path / "fake_black_sea.zarr"
    ds.to_zarr(path_data, mode="w")
    return path_data

def test_from_tensor_to_indices(fake_black_sea_dataset):
    """Testing if indices are extracted properly."""
    mappings = [
        from_tensor_to_indices(path=fake_black_sea_dataset, variables=variables, region=FAKE_REGION)
        for variables in [
            ["CHL", "DOX"],
            ["votemper", "ssh"],
            ["ssh", "CHL", "rho"],
        ]
    ]
    assert mappings[0] == {
        "CHL": (0, 4),
        "DOX": (4, 8),
    }, f"ERROR - Expected indices for CHL and DOX are incorrect, got {mappings[0]}"

    assert mappings[1] == {
        "votemper": (0, 4),
        "ssh": (4, 5),
    }, f"ERROR - Expected indices for votemper and ssh are incorrect, got {mappings[1]}"

    assert mappings[2] == {
        "ssh": (0, 1),
        "CHL": (1, 5),
        "rho": (5, 9),
    }, f"ERROR - Expected indices for ssh, CHL, and rho are incorrect, got {mappings[2]}"



def test_from_tensor_to_xarray(fake_black_sea_dataset):
    """Testing if dataset variables are correctly retrieved."""

    # Be careful to load dataset with variables in the same order as the variables list
    variables = ["CHL", "DOX", "rho", "vosaline", "votemper", "ssh"]
    tensor_dataset = torch.as_tensor((
        xr.open_zarr(fake_black_sea_dataset)[variables]
        .to_stacked_array(new_dim="z_total", sample_dims=("time", "longitude", "latitude"))
        .transpose("z_total", "time", ...)
    ).load().data)

    xarray = from_tensor_to_xarray(
        x=tensor_dataset,
        path=fake_black_sea_dataset,
        variables=variables,
        region=FAKE_REGION,
    )

    v1 = torch.from_numpy(xarray["CHL"].data)
    v2 = torch.from_numpy(xarray["DOX"].data)
    v3 = torch.from_numpy(xarray["rho"].data)
    v4 = torch.from_numpy(xarray["vosaline"].data)
    v5 = torch.from_numpy(xarray["votemper"].data)
    v6 = torch.from_numpy(xarray["ssh"].data)

    assert torch.allclose(v1, torch.ones_like(v1) * 1, atol=1e-1)
    assert torch.allclose(v2, torch.ones_like(v2) * 2, atol=1e-1)
    assert torch.allclose(v3, torch.ones_like(v3) * 3, atol=1e-1)
    assert torch.allclose(v4, torch.ones_like(v4) * 4, atol=1e-1)
    assert torch.allclose(v5, torch.ones_like(v5) * 5, atol=1e-1)
    assert torch.allclose(v6, torch.ones_like(v6) * 6, atol=1e-1)
