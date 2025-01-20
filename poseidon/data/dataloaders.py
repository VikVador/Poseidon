r"""Dataloaders."""

from torch.utils.data import DataLoader
from typing import Any, Callable, Optional, Sequence, Tuple

# isort: split
from poseidon.data.datasets import (
    PoseidonDataset,
    get_datasets,
    get_toy_datasets,
)


def infinite_dataloader(dataloader: DataLoader, steps: int) -> Any:
    r"""Transforms a PyTorch dataloader into an infinite dataloader."""
    s = 1
    while s < (steps + 1):
        for batch in dataloader:
            yield s, batch
            s += 1


def get_dataloaders(**kwargs) -> Tuple[DataLoader, DataLoader, DataLoader]:
    r"""Returns the training, validation, and test dataloaders.

    Region:
        Black Sea.

    Shuffling:
        Only the training dataset is shuffled (by default).

    Splits:
        Training: 1995-01-01 to 2017-12-31.
        Validation: 2018-01-01 to 2020-12-31.
        Test: 2021-01-01 to 2022-12-31.

    Arguments:
        trajectory_size: Number of time steps in each sample.
        variables: Variable names to retain from the dataset.
        shuffle: List of booleans defining which dataset to shuffle.
        infinite: Whether to transform dataloaders as infinite iterators or not.
        steps: If infinite, the maximum number of steps to iterate.
        kwargs: Keyword arguments passed to the dataloader.
    """
    return _get_dataloaders_from_datasets(
        get_datasets=get_datasets,
        **kwargs,
    )


def get_toy_dataloaders(**kwargs) -> Tuple[DataLoader, DataLoader, DataLoader]:
    r"""Returns the toy training, validation, and test dataloaders.

    Region:
        Black Sea Continental Shelf.

    Shuffling:
        Only the training dataset is shuffled (by default).

    Splits:
        Training: 1995-01-01 to 2017-12-31.
        Validation: 2018-01-01 to 2020-12-31.
        Test: 2021-01-01 to 2022-12-31.

    Arguments:
        trajectory_size: Number of time steps in each sample.
        variables: Variable names to retain from the dataset.
        shuffle: List of booleans defining which dataset to shuffle.
        infinite: Whether to transform dataloaders as infinite iterators or not.
        steps: If infinite, the maximum number of steps to iterate.
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
    infinite: bool = False,
    steps: Optional[int] = None,
    **kwargs: Any,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    r"""Helper tool to generate dataloaders from datasets.

    Arguments:
        get_datasets: A function that returns datasets.
    """

    if infinite:
        assert (
            steps is not None
        ), "ERROR - Maximum number of steps needed to create an 'infinite' dataloaders."

    datasets = get_datasets(
        trajectory_size=trajectory_size,
        variables=variables,
    )

    dataloaders = [
        DataLoader(
            dataset,
            shuffle=shuffle[i],
            pin_memory=True,
            **kwargs,
        )
        for i, dataset in enumerate(datasets)
    ]

    if infinite:
        dataloaders = [infinite_dataloader(dataloader, steps) for dataloader in dataloaders]

    return tuple(dataloaders)
