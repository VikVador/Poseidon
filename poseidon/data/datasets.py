r"""Datasets."""

import dask
import torch
import xarray as xr

from pathlib import Path
from torch import Tensor
from torch.utils.data import Dataset
from typing import Dict, Optional, Sequence, Tuple

# isort: split
from poseidon.config import PATH_DATA
from poseidon.data.const import (
    DATASET_DATES_TEST,
    DATASET_DATES_TRAINING,
    DATASET_DATES_VALIDATION,
    DATASET_NAN_FILL,
    DATASET_REGION,
    TOY_DATASET_DATES_TEST,
    TOY_DATASET_DATES_TRAINING,
    TOY_DATASET_DATES_VALIDATION,
    TOY_DATASET_REGION,
)
from poseidon.date import assert_date_format, get_date_features


class PoseidonDataset(Dataset):
    r"""Creates a Poseidon dataset.

    Arguments:
        path: Path to the Zarr dataset.
        start_date: Start date of the data split (format: 'YYYY-MM-DD').
        end_date: End date of the data split (format: 'YYYY-MM-DD').
        trajectory_size: Number of time steps in each sample.
        variables: Variable names to retain from the preprocessed dataset.
        region: Region of interest to extract from the dataset.
    """

    def __init__(
        self,
        path: Path,
        start_date: str,
        end_date: str,
        trajectory_size: int = 1,
        variables: Optional[Sequence[str]] = None,
        region: Optional[Dict[str, Tuple[int, int]]] = None,
    ):
        super().__init__()

        assert_date_format(start_date)
        assert_date_format(end_date)
        self.dataset = xr.open_zarr(path).sel(time=slice(start_date, end_date))
        self.dataset = self.dataset[variables] if variables else self.dataset
        self.dataset = self.dataset.isel(**region) if region else self.dataset
        self.trajectory_size = trajectory_size

    def __len__(self) -> int:
        r"""Return the total number of samples in the dataset."""
        return self.dataset.time.size - self.trajectory_size + 1

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        r"""Gets and preprocesses a sample from the dataset.

        Argument:
            idx: Index of the first time step in the sample.

        Returns:
            sample: Preprocessed data tensor of shape (trajectory_size, z_total, x, y).
            time: Date features corresponding to each day of the trajectory.
        """

        return self.preprocess(idx, idx + self.trajectory_size)

    def preprocess(self, step_start: int, step_end: int) -> Tuple[Tensor, Tensor]:
        r"""Extract and reshape a sample from the dataset.

        Arguments:
            step_start: Start index of the sample.
            step_end: End index of the sample.

        Returns:
            sample: A tensor containing the preprocessed sample.
            time: Date features corresponding to each day of the trajectory.
        """

        # Handle large data by splitting into smaller chunks
        with dask.config.set(**{"array.slicing.split_large_chunks": True}):
            sample = self.dataset.isel(time=slice(step_start, step_end))
            sample = sample.fillna(DATASET_NAN_FILL)
            time = [get_date_features(sample.time[i].values) for i in range(sample.time.size)]
            time = torch.stack(time, dim=0)
            sample = sample.to_stacked_array(
                new_dim="z_total", sample_dims=("time", "longitude", "latitude")
            ).transpose("z_total", "time", ...)

        return torch.as_tensor(sample.load().data), time


def get_datasets(**kwargs) -> Tuple[PoseidonDataset, PoseidonDataset, PoseidonDataset]:
    r"""Returns the training, validation, and test datasets.

    Splits:
        Training: 1995-01-01 to 2015-12-31.
        Validation: 2016-01-01 to 2019-12-31.
        Test: 2020-01-01 to 2022-12-31.

    Arguments:
        kwargs: Keyword arguments passed to :class:`PoseidonDataset`.
    """

    datasets = [
        PoseidonDataset(
            path=PATH_DATA,
            start_date=start_date,
            end_date=end_date,
            region=DATASET_REGION,
            **kwargs,
        )
        for start_date, end_date in [
            DATASET_DATES_TRAINING,
            DATASET_DATES_VALIDATION,
            DATASET_DATES_TEST,
        ]
    ]

    return tuple(datasets)


def get_toy_datasets(
    variables: Optional[Sequence[str]] = None,
    **kwargs,
) -> Tuple[PoseidonDataset, PoseidonDataset, PoseidonDataset]:
    r"""Returns the toy training, validation, and test datasets.

    Variables:
        Only the sea surface height, temperature and oyxgen fields.

    Splits:
        Training: 2014-01-01 to 2015-12-31.
        Validation: 2019-01-01 to 2019-12-31.
        Test: 2022-01-01 to 2022-12-31.

    Arguments:
        variables: Variable names to retain from the dataset.
        kwargs: Keyword arguments passed to :class:`PoseidonDataset`.
    """

    if variables is None:
        variables = ["ssh", "votemper", "DOX"]

    datasets = [
        PoseidonDataset(
            path=PATH_DATA,
            start_date=start_date,
            end_date=end_date,
            variables=variables,
            region=TOY_DATASET_REGION,
            **kwargs,
        )
        for start_date, end_date in [
            TOY_DATASET_DATES_TRAINING,
            TOY_DATASET_DATES_VALIDATION,
            TOY_DATASET_DATES_TEST,
        ]
    ]

    return tuple(datasets)
