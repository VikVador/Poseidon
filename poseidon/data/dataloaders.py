r"""Dataloaders."""

from torch.utils.data import DataLoader
from typing import (
    Any,
    Optional,
    Sequence,
    Tuple,
)

# isort: split
from poseidon.data.datasets import get_datasets


def infinite_dataloader(
    dataloader: DataLoader,
    steps: int,
) -> Any:
    r"""Makes a basic PyTorch dataloader 'infinite'.

    Arguments:
        dataloader: A PyTorch dataloader.
        steps: Maximum number of iterations before loop stops.
    """
    for _ in range(steps):
        for batch in dataloader:
            yield batch
            steps -= 1
            if steps <= 0:
                return


def get_dataloaders(
    trajectory_size: int = 1,
    variables: Optional[Sequence[str]] = None,
    shuffle: Tuple[bool, bool, bool] = (True, False, False),
    linspace: Optional[Sequence[bool]] = [False, False, False],
    linspace_samples: Optional[Sequence[int]] = [None, None, None],
    infinite: Optional[Sequence[bool]] = [False, False, False],
    steps: Optional[Sequence[int]] = [None, None, None],
    **kwargs: Any,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    r"""Returns the training, validation, and test dataloaders.

    Shuffling:
        Only the training dataset is shuffled (by default).

    Arguments:
        trajectory_size: Number of time steps in trajectory.
        variables: Variable names to retain from the dataset.
        shuffle: List of booleans defining which dataset to shuffle.
        linspace: Whether to extract samples at linearly spaced intervals.
        linspace_samples: Number of linearly spaced samples to extract, if `linspace` is True.
        infinite: Whether to transform dataloaders as infinite iterators or not.
        steps: If infinite, the maximum number of steps to iterate.
        kwargs: Keyword arguments passed to the dataloader.
    """

    for inf, stp in zip(infinite, steps):
        if inf:
            assert (
                stp is not None
            ), "ERROR - Maximum number of iterations needed to create an 'infinite' dataloader."

    for lin, lin_s in zip(linspace, linspace_samples):
        if lin:
            assert (
                lin_s is not None
            ), "ERROR - Number of samples `linspace_samples` needed to create a 'linspace' dataloader"

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
