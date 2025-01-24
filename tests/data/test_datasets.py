r"""Tests for the poseidon.data.datasets module."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pathlib import Path

# isort: split
from poseidon.data.datasets import PoseidonDataset

LINSPACE_SAMPLES, TRAJECTORY_SIZE = (
    np.random.randint(1, 10),
    np.random.randint(1, 3),
)

FAKE_REGION = {
    "latitude": slice(0, 2),
    "longitude": slice(0, 3),
    "level": slice(0, 4),
}


# fmt: off
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
@pytest.mark.parametrize("region", [None, FAKE_REGION])
def test_PoseidonDataset(fake_black_sea_dataset, trajectory_size, variables, region):
    r"""Testing PoseidonDataset class."""

    ds = PoseidonDataset(
        path=fake_black_sea_dataset,
        date_start="1995-01-01",
        date_end="1995-01-10",
        trajectory_size=trajectory_size,
        variables=variables,
        region=region,
    )

    # Dataset
    SAMPLES   = len(ds)
    SAMPLES_E = ds.dataset.time.size - trajectory_size + 1

    # Sample
    x, time = ds[0]

    Z, T, LAT, LON = x.shape

    Z_E   = 21 if variables is None else 5
    T_E   = trajectory_size
    LAT_E = 4 if region is None else 2
    LON_E = 4 if region is None else 3

    # Time
    TIME_T     = time.shape[0]
    TIME_DATES = time.shape[1]

    TIME_T_E     = trajectory_size
    TIME_DATES_E = 4

    # Assertion
    assert Z == Z_E,                   f"ERROR - Wrong number of total levels ({Z_E} != {Z})"
    assert T == T_E,                   f"ERROR - Wrong number of time steps ({T_E} != {T})"
    assert LAT == LAT_E,               f"ERROR - Wrong number of latitude points ({LAT_E} != {LAT})"
    assert LON == LON_E,               f"ERROR - Wrong number of longitude points ({LON_E} != {LON})"
    assert TIME_T == TIME_T_E,         f"ERROR - Wrong number of time steps ({TIME_T_E} != {TIME_T})"
    assert TIME_DATES == TIME_DATES_E, f"ERROR - Wrong number of elements in date({TIME_DATES_E} != {TIME_DATES})"
    assert SAMPLES == SAMPLES_E,       f"ERROR - Wrong number of samples in dataset ({SAMPLES_E} != {SAMPLES})"


@pytest.mark.parametrize("linspace", [True, False])
def test_PoseidonDataset_linspace(fake_black_sea_dataset, linspace):
    r"""Testing PoseidonDataset class linspace."""

    ds = PoseidonDataset(
        path=fake_black_sea_dataset,
        date_start="1995-01-01",
        date_end="1995-01-10",
        trajectory_size=TRAJECTORY_SIZE,
        variables=["ssh", "votemper"],
        region=FAKE_REGION,
        linspace=linspace,
        linspace_samples=LINSPACE_SAMPLES,
    )

    # Dataset
    SAMPLES   = len(ds)
    SAMPLES_E = LINSPACE_SAMPLES if linspace else ds.dataset.time.size - TRAJECTORY_SIZE + 1

    # Assertion
    assert SAMPLES == SAMPLES_E, f"ERROR - Wrong number of samples in dataset ({SAMPLES_E} != {SAMPLES})"
