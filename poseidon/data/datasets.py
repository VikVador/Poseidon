r"""Datasets."""

import dask
import torch
import xarray as xr

from pathlib import Path
from torch import Tensor
from torch.utils.data import Dataset
from typing import (
    Dict,
    Optional,
    Sequence,
    Tuple,
)

# isort: split
from poseidon.config import PATH_DATA
from poseidon.data.const import (
    DATASET_DATES_TEST,
    DATASET_DATES_TRAINING,
    DATASET_DATES_VALIDATION,
    DATASET_REGION,
    DATASET_VARIABLES,
    LAND_VALUE,
    TOY_DATASET_DATES_TEST,
    TOY_DATASET_DATES_TRAINING,
    TOY_DATASET_DATES_VALIDATION,
    TOY_DATASET_REGION,
    TOY_DATASET_VARIABLES,
)
from poseidon.data.tools import (
    assert_date_format,
    get_date_features,
)


class PoseidonDataset(Dataset):
    r"""Creates a :class:`PoseidonDataset`.

    Arguments:
        path: Path to the Zarr dataset.
        date_start: Start date of the data split (format: 'YYYY-MM-DD').
        date_end: End date of the data split (format: 'YYYY-MM-DD').
        variables: Variable names to retain from the dataset.
        trajectory_size: Number of time steps in trajectory
        linspace: Whether to extract samples at linearly spaced intervals.
        linspace_samples: Number of linearly spaced samples to extract, if `linspace` is True.
        region: Region of interest to extract from the dataset.
    """

    def __init__(
        self,
        path: Path,
        date_start: str,
        date_end: str,
        variables: Sequence[str],
        trajectory_size: int = 1,
        linspace: Optional[bool] = False,
        linspace_samples: Optional[int] = None,
        region: Optional[Dict[str, Tuple[int, int]]] = None,
    ):
        super().__init__()

        assert_date_format(date_start)
        assert_date_format(date_end)

        self.dataset = xr.open_zarr(path).sel(time=slice(date_start, date_end))
        self.dataset = self.dataset[variables] if variables else self.dataset
        self.dataset = self.dataset.isel(**region) if region else self.dataset

        self.trajectory_size = trajectory_size
        self.linspace = linspace
        self.linspace_samples = linspace_samples
        self.linspace_samples_index = None

        # Checking if linearly spaced samples are requested
        if self.linspace:
            self.linspace_samples_index = torch.linspace(
                0,
                self.dataset.time.size - self.trajectory_size,
                linspace_samples,
                dtype=torch.int64,
            )

    def __len__(self) -> int:
        r"""Return the total number of samples in the dataset."""
        return (
            self.linspace_samples
            if self.linspace
            else (self.dataset.time.size - self.trajectory_size + 1)
        )

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        r"""Gets and preprocesses a sample from the dataset."""
        idx = self.linspace_samples_index[idx].item() if self.linspace else idx
        return self.preprocess(idx, idx + self.trajectory_size)

    def preprocess(
        self,
        step_start: int,
        step_end: int,
    ) -> Tuple[Tensor, Tensor]:
        r"""Extract and reshape a sample from the dataset.

        Arguments:
            step_start: Start index of the sample.
            step_end: End index of the sample.

        Returns:
            sample: A trajectory sample.
            time: Date features corresponding to each day of the trajectory.
        """

        with dask.config.set(**{"array.slicing.split_large_chunks": True}):
            sample = self.dataset.isel(time=slice(step_start, step_end))
            sample = sample.fillna(LAND_VALUE)
            time = [get_date_features(sample.time[i].values) for i in range(sample.time.size)]
            time = torch.stack(time, dim=0)
            sample = sample.to_stacked_array(
                new_dim="z_total", sample_dims=("time", "longitude", "latitude")
            ).transpose("z_total", "time", ...)
            sample = torch.as_tensor(sample.load().data.copy())

        return sample.to(dtype=torch.float32), time.to(dtype=torch.float32)


def get_toy_datasets(
    variables: Optional[Sequence[str]] = None,
    linspace: Optional[Sequence[bool]] = [False, False, False],
    linspace_samples: Optional[Sequence[int]] = [None, None, None],
    **kwargs,
) -> Tuple[PoseidonDataset, PoseidonDataset, PoseidonDataset]:
    r"""Returns the toy training, validation, and test :class:`PoseidonDataset`.

    Region:
        Black Sea Continental Shelf (Debugging).

    Splits:
        Training: 2015-01-01 to 2017-12-31.
        Validation: 2020-01-01 to 2020-12-31.
        Test: 2022-01-01 to 2022-12-31.

    Arguments:
        variables: Variable names to retain from the dataset.
        linspace: Whether to extract samples at linearly spaced intervals.
        linspace_samples: Number of linearly spaced samples to extract, if `linspace` is True.
        kwargs: Keyword arguments passed to :class:`PoseidonDataset`.
    """

    if variables is None:
        variables = TOY_DATASET_VARIABLES

    datasets = [
        PoseidonDataset(
            path=PATH_DATA,
            date_start=date_start,
            date_end=date_end,
            region=TOY_DATASET_REGION,
            variables=variables,
            linspace=l,
            linspace_samples=s,
            **kwargs,
        )
        for (date_start, date_end), l, s in zip(
            [
                TOY_DATASET_DATES_TRAINING,
                TOY_DATASET_DATES_VALIDATION,
                TOY_DATASET_DATES_TEST,
            ],
            linspace,
            linspace_samples,
        )
    ]

    return tuple(datasets)


def get_datasets(
    variables: Optional[Sequence[str]] = None,
    linspace: Optional[Sequence[bool]] = [False, False, False],
    linspace_samples: Optional[Sequence[int]] = [None, None, None],
    **kwargs,
) -> Tuple[PoseidonDataset, PoseidonDataset, PoseidonDataset]:
    r"""Returns the training, validation, and test :class:`PoseidonDataset`.

    Region:
        Black Sea Continental Shelf.

    Splits:
        Training: 1995-01-01 to 2017-12-31.
        Validation: 2018-01-01 to 2020-12-31.
        Test: 2021-01-01 to 2022-12-31.

    Arguments:
        variables: Variable names to retain from the dataset.
        linspace: Whether to extract samples at linearly spaced intervals.
        linspace_samples: Number of linearly spaced samples to extract, if `linspace` is True.
        kwargs: Keyword arguments passed to :class:`PoseidonDataset`.
    """

    if variables is None:
        variables = DATASET_VARIABLES

    datasets = [
        PoseidonDataset(
            path=PATH_DATA,
            date_start=date_start,
            date_end=date_end,
            region=DATASET_REGION,
            variables=variables,
            linspace=l,
            linspace_samples=s,
            **kwargs,
        )
        for (date_start, date_end), l, s in zip(
            [
                DATASET_DATES_TRAINING,
                DATASET_DATES_VALIDATION,
                DATASET_DATES_TEST,
            ],
            linspace,
            linspace_samples,
        )
    ]

    return tuple(datasets)
