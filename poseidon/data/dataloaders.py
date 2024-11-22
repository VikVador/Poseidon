r"""Dataloaders."""

from torch.utils.data import DataLoader
from typing import Any, Callable, Optional, Sequence, Tuple

# isort: split
from poseidon.data.datasets import (
    PoseidonDataset,
    get_datasets,
    get_toy_datasets,
)


def get_dataloaders(**kwargs) -> Tuple[DataLoader, DataLoader, DataLoader]:
    r"""Returns the training, validation, and test dataloaders.

    Shuffling:
        Only the training dataset is shuffled (by default).

    Splits:
        Training: 1995-01-01 to 2015-12-31.
        Validation: 2016-01-01 to 2019-12-31.
        Test: 2020-01-01 to 2022-12-31.

    Arguments:
        trajectory_size: Number of time steps in each sample.
        variables: Variable names to retain from the dataset.
        shuffle: List of booleans defining which dataset to shuffle
        kwargs: Keyword arguments passed to the dataloader.
    """
    return _get_dataloaders_from_datasets(
        get_datasets=get_datasets,
        **kwargs,
    )


def get_toy_dataloaders(**kwargs) -> Tuple[DataLoader, DataLoader, DataLoader]:
    r"""Returns the toy training, validation, and test dataloaders.

    Variables:
        Only the sea surface height, temperature and oyxgen fields.

    Shuffling:
        Only the training dataset is shuffled (by default).

    Splits:
        Training: 2014-01-01 to 2015-12-31.
        Validation: 2019-01-01 to 2019-12-31.
        Test: 2022-01-01 to 2022-12-31.

    Arguments:
        trajectory_size: Number of time steps in each sample.
        variables: Variable names to retain from the dataset.
        shuffle: List of booleans defining which dataset to shuffle.
        kwargs: Keyword arguments passed to the dataloader.
    """
    return _get_dataloaders_from_datasets(
        get_datasets=get_toy_datasets,
        **kwargs,
    )


def _get_dataloaders_from_datasets(
    get_datasets: Callable[..., Tuple[PoseidonDataset, PoseidonDataset, PoseidonDataset]],
    trajectory_size: int = 1,
    variables: Optional[Sequence[str]] = None,
    shuffle: Tuple[bool, bool, bool] = (True, False, False),
    **kwargs: Any,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    r"""Helper tool to generate dataloaders from datasets.

    Arguments:
        get_datasets: A function that returns datasets.
    """

    datasets = get_datasets(
        trajectory_size=trajectory_size,
        variables=variables,
    )

    dataloaders = [
        DataLoader(
            dataset,
            shuffle=shuffle[i],
            **kwargs,
        )
        for i, dataset in enumerate(datasets)
    ]

    return tuple(dataloaders)
