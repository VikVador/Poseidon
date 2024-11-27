"""Tests for the poseidon.data.statistics module."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pathlib import Path

# isort: split
from poseidon.data.statistics import PoseidonStatistics


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


@pytest.fixture
def fake_black_sea_statistics(tmp_path) -> Path:
    r"""Create a fake Black Sea statistics dataset.

    The dataset contains mean values for several variables over specified levels.

    Arguments:
        tmp_path: Path to a temporary directory provided by pytest.

    Returns:
        Path to the created Zarr dataset representing a Black Sea statistics dataset.
    """
    level = np.array([0, 1, 2, 2], dtype=np.int64)

    variables = {
        "CHL": 1,
        "DOX": 2,
        "rho": 3,
        "vosaline": 5,
        "votemper": 6,
    }

    # Assign mean values to levels
    data_vars = {
        name: (["level"], np.full_like(level, value)) for name, value in variables.items()
    }
    data_vars["ssh"] = ([], np.float64(4))

    ds = xr.Dataset(data_vars, coords={"level": level})
    path_data = tmp_path / "fake_black_sea_statistics.zarr"
    ds.to_zarr(path_data, mode="w")
    return path_data


def test_PoseidonStatistics(fake_black_sea_dataset, fake_black_sea_statistics):
    r"""Testing PoseidonStatistics class."""

    stats_calculator = PoseidonStatistics()

    dataset = xr.open_zarr(fake_black_sea_dataset)
    dataset_statistics = xr.open_zarr(fake_black_sea_statistics).load()

    stats_calculator.update(dataset)
    mean = stats_calculator.get_mean().load()

    # Validating statistics
    xr.testing.assert_allclose(mean, dataset_statistics, rtol=0.1, atol=0.1)
