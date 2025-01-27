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
    r"""Transforms a basic PyTorch dataloader into an 'infinite' dataloader.

    Arguments:
        dataloader: A PyTorch :class:`dataloader`.
        steps: Maximum number of steps to iterate before infinite loop stops.
    """
    for _ in range(steps):
        for batch in dataloader:
            yield batch
            steps -= 1
            if steps <= 0:
                return


def get_dataloaders(**kwargs) -> Tuple[DataLoader, DataLoader, DataLoader]:
    r"""Returns the training, validation, and test dataloaders.

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
        linspace: Whether to extract samples at linearly spaced intervals.
        linspace_samples: Number of linearly spaced samples to extract, if `linspace` is True.
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
        Black Sea Continental Shelf (Debugging).

    Shuffling:
        Only the training dataset is shuffled (by default).

    Splits:
        Training: 2017-01-01 to 2017-12-31.
        Validation: 2020-01-01 to 2020-12-31.
        Test: 2022-01-01 to 2022-12-31.

    Arguments:
        trajectory_size: Number of time steps in each sample.
        variables: Variable names to retain from the dataset.
        shuffle: List of booleans defining which dataset to shuffle.
        linspace: Whether to extract samples at linearly spaced intervals.
        linspace_samples: Number of linearly spaced samples to extract, if `linspace` is True.
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
    linspace: Optional[Sequence[bool]] = [False, False, False],
    linspace_samples: Optional[Sequence[int]] = [None, None, None],
    infinite: Optional[Sequence[bool]] = [False, False, False],
    steps: Optional[Sequence[int]] = [None, None, None],
    **kwargs: Any,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    r"""Helper tool to generate dataloaders from datasets.

    Arguments:
        get_datasets: A function that returns datasets.
    """

    # Securities
    for inf, stp in zip(infinite, steps):
        if inf:
            assert (
                stp is not None
            ), "ERROR - Maximum number of steps needed to create an 'infinite' dataloaders."

    for lin, lin_s in zip(linspace, linspace_samples):
        if lin:
            assert (
                lin_s is not None
            ), "ERROR - Number of samples needed to create a 'linspace' dataloaders"

    datasets = get_datasets(
        trajectory_size=trajectory_size,
        variables=variables,
        linspace=linspace,
        linspace_samples=linspace_samples,
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

    dataloaders = [
        infinite_dataloader(dl, st) if inf else dl
        for inf, st, dl in zip(infinite, steps, dataloaders)
    ]

    return tuple(dataloaders)
