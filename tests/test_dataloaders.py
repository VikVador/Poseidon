"""Tests for the aang.data.dataloaders module."""

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from pathlib import Path
from torch import Tensor
from typing import Optional, Sequence

# isort: split
from poseidon.data.dataloaders import _get_dataloaders_from_datasets
from poseidon.data.datasets import PoseidonDataset

torch.manual_seed(1)


# fmt: off
@pytest.fixture
def fake_black_sea_dataset(tmp_path) -> Path:
    r"""Create a fake Black Sea dataset.

    Arguments:
        tmp_path: Path to a temporary directory provided by pytest.

    Returns:
        Path to the created Zarr dataset representing a toy Black Sea dataset.
    """
    time = pd.date_range("1995-01-01", periods=90, freq="d")
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
@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("shuffle", [[True, False, False], [False, True, True]])
def test_get_dataloaders(fake_black_sea_dataset, trajectory_size, variables, batch_size, shuffle):
    r"""Testing generation of dataloaders."""

    def get_fake_datasets(trajectory_size: int, variables: Optional[Sequence[str]] = None):
        """Helper callabale to generate fake datasets."""
        return [
            PoseidonDataset(
                path=fake_black_sea_dataset,
                start_date=ds,
                end_date=de,
                trajectory_size=trajectory_size,
                variables=variables,
            )
            for (ds, de) in zip(
                ["1995-01-01", "1995-02-01", "1995-03-01"],
                ["1995-01-31", "1995-02-28", "1995-03-31"],
            )
        ]

    train_dl, val_dl, test_dl = _get_dataloaders_from_datasets(
        get_datasets=get_fake_datasets,
        trajectory_size=trajectory_size,
        variables=variables,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    x_train, t_train = next(iter(train_dl))
    x_val, t_val = next(iter(val_dl))
    x_test, t_test = next(iter(test_dl))

    # Expected dimensions
    B_E, Z_E, T_E, LAT_E, LON_E, TIME_B_E, TIME_T_E, TIME_DAYS_E = (
        batch_size,
        21 if variables is None else 5,
        trajectory_size,
        4,
        4,
        batch_size,
        trajectory_size,
        3,
    )

    init_dates = [
        Tensor([1, 1, 0]),
        Tensor([2, 1, 0]),
        Tensor([3, 1, 0])
    ]

    for x, t, s, d in zip([x_train, x_val, x_test], [t_train, t_val, t_test], shuffle, init_dates):

        B, Z, T, LAT, LON = x.shape
        TIME_B, TIME_T, TIME_DAYS = t.shape

        assert B == B_E,                 f"ERROR - Wrong number of batch size ({B_E} != {B})"
        assert Z == Z_E,                 f"ERROR - Wrong number of total levels ({Z_E} != {Z})"
        assert T == T_E,                 f"ERROR - Wrong number of time steps ({T_E} != {T})"
        assert LAT == LAT_E,             f"ERROR - Wrong number of latitude points ({LAT_E} != {LAT})"
        assert LON == LON_E,             f"ERROR - Wrong number of longitude points ({LON_E} != {LON})"
        assert TIME_B == TIME_B_E,       f"ERROR - Wrong number of batch size ({TIME_B_E} != {TIME_B})"
        assert TIME_T == TIME_T_E,       f"ERROR - Wrong number of time steps ({TIME_T_E} != {TIME_T})"
        assert TIME_DAYS == TIME_DAYS_E, f"ERROR - Wrong number of elements in date ({TIME_DAYS_E} != {TIME_DAYS})"

        # Since dataset is small, we need to fix the seed to check for shuffling
        if s:
            assert not torch.allclose(t[0, 0], d.to(dtype=torch.int64)), \
            "ERROR - Dataset is not shuffled."
