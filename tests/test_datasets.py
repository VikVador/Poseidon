r"""Tests for the aang.data.dataset module."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pathlib import Path

# isort: split
from poseidon.data.datasets import PoseidonDataset

TOY_REGION = {
    "latitude": slice(0, 2),
    "longitude": slice(0, 3),
    "level": slice(0, 4),
}


@pytest.fixture
def fake_black_sea_dataset(tmp_path) -> Path:
    r"""Create a fake Black Sea dataset.

    Arguments:
        tmp_path: Path to a temporary directory provided by pytest.

    Returns:
        Path to the created Zarr dataset representing a toy Black Sea dataset.
    """
    time = pd.date_range("1995-01-01", periods=12, freq="d")
    latitude = np.array([0, 1, 2, 3], dtype=np.float32)
    longitude = np.array([0, 1, 2, 3], dtype=np.float32)
    level = np.array([0, 1, 2, 2], dtype=np.int64)

    fluctuation = np.random.randn(*map(len, [time, level, latitude, longitude]))

    chl = 1 + 0.1 * fluctuation
    dox = 2 + 0.1 * fluctuation
    rho = 3 + 0.1 * fluctuation
    ssh = 4 + 0.1 * fluctuation
    sal = 5 + 0.1 * fluctuation
    tmp = 6 + 0.1 * fluctuation

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


@pytest.mark.parametrize("trajectory_size", [1, 3])
@pytest.mark.parametrize("variables", [None, ["ssh", "votemper"]])
@pytest.mark.parametrize("region", [None, TOY_REGION])
def test_PoseidonDataset(fake_black_sea_dataset, trajectory_size, variables, region):
    r"""Testing PoseidonDataset class."""

    ds = PoseidonDataset(
        path=fake_black_sea_dataset,
        start_date="1995-01-01",
        end_date="1995-01-10",
        trajectory_size=trajectory_size,
        variables=variables,
        region=region,
    )

    x, time = ds[0]
    z_total, t, lat, lon = x.shape
    lt, ln = (4, 4) if region is None else (2, 3)
    zt = (5 * 4 + 1) if variables is None else (1 * 4 + 1)

    assert t == trajectory_size, f"Expected {trajectory_size} but got {t}"
    assert lat == lt, f"Expected {lt} but got {lat}"
    assert lon == ln, f"Expected {ln} but got {lon}"
    assert z_total == zt, f"Expected {zt} but got {z_total}"
    assert len(ds) == ds.dataset.time.size - trajectory_size + 1, (
        f"Expected {ds.dataset.time.size - trajectory_size + 1} " f"but got {len(ds)}"
    )
    assert time.shape == (
        trajectory_size,
        3,
    ), f"Expected ({trajectory_size}, 3) but got {time.shape}"
