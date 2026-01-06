r"""Dataloaders."""

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
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
    is_distributed: bool = False,
) -> Any:
    r"""Makes a basic PyTorch dataloader 'infinite'.

    Arguments:
        dataloader: A PyTorch dataloader.
        steps: Maximum number of iterations.
        is_distributed: Whether running in distributed mode.
    """
    epoch = 0
    steps_remaining = steps

    while steps_remaining > 0:
        # Set epoch for proper shuffling in DDP
        if is_distributed and hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)

        for batch in dataloader:
            yield batch
            steps_remaining -= 1
            if steps_remaining <= 0:
                return

        epoch += 1


def get_dataloaders(
    trajectory_size: int = 1,
    variables: Optional[Sequence[str]] = None,
    shuffle: Tuple[bool, bool, bool] = (True, False, False),
    linspace: Optional[Sequence[bool]] = [False, False, False],
    linspace_samples: Optional[Sequence[int]] = [None, None, None],
    infinite: Optional[Sequence[bool]] = [False, False, False],
    steps: Optional[Sequence[int]] = [None, None, None],
    rank: int = 0,
    world_size: int = 1,
    is_distributed: bool = False,
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
        rank: Rank of current process in distributed training.
        world_size: Total number of processes in distributed training.
        is_distributed: Whether running in distributed mode.
        kwargs: Keyword arguments passed to the dataloader.
    """

    for inf, stp in zip(infinite, steps):
        if inf:
            assert stp is not None, "ERROR - Maximum number of iterations needed to create an 'infinite' dataloader."

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

    # Extract and rename batch_size_per_gpu to batch_size for DataLoader
    dataloader_kwargs = kwargs.copy()
    batch_size = dataloader_kwargs.pop("batch_size_per_gpu", dataloader_kwargs.pop("batch_size", 1))

    dataloaders = []
    for i, dataset in enumerate(datasets):
        if is_distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle[i],
                drop_last=True,
            )
            dataloader_shuffle = False
        else:
            sampler = None
            dataloader_shuffle = shuffle[i]

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,  # TO DO: Handles non-DPP training with batch size per gpu
            sampler=sampler,
            shuffle=dataloader_shuffle,
            pin_memory=True,
            persistent_workers=dataloader_kwargs.get("num_workers", 0) > 0,
            **dataloader_kwargs,
        )

        dataloaders.append(dataloader)

    # Handle infinite dataloaders
    dataloaders = [
        infinite_dataloader(dl, st, is_distributed) if inf else dl for inf, st, dl in zip(infinite, steps, dataloaders)
    ]

    return tuple(dataloaders)
