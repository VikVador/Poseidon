r"""Poseidon - Tools to perform the training of a denoiser."""

import torch

from torch.optim import Adam
from typing import Dict, Tuple

# isort: split
from poseidon.config import POSEIDON_MODEL
from poseidon.data.const import DATASET_REGION, TOY_DATASET_REGION
from poseidon.data.dataloaders import get_dataloaders, get_toy_dataloaders
from poseidon.diffusion.backbone import PoseidonBackbone
from poseidon.diffusion.denoiser import PoseidonDenoiser
from poseidon.diffusion.loss import PoseidonLoss
from poseidon.diffusion.noise import PoseidonNoiseSchedule
from poseidon.network.save import generate_model_name, save_configurations, save_model


def time_tokenizer(data: torch.Tensor) -> torch.Tensor:
    r"""Tokenizes a tensor containing temporal information.

    Arguments:
        data: Time tensor of shape (batch_size, 3), i.e. month, day, and hour.
    """
    months = data[:, 0] - 1
    days = data[:, 1] - 1
    hour_mapping = {6: 0, 12: 1, 18: 2, 24: 3}
    hours = torch.tensor([hour_mapping[int(hour)] for hour in data[:, 2]], dtype=torch.long)
    return torch.stack([months, days, hours], dim=1)


def extract_blankets_in_trajectories(
    x: torch.Tensor, blanket_idx: Tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    r"""Extracts various blankets from a given batch of trajectories.

    Arguments:
        x: Input tensor (B, T, C, H, W).
        blanket_idx: Tuple containing the starting and ending indices of the blankets.

    Returns:
        blankets: Blanket tensor of shape (B, 2k+1, C, H, W).
    """
    idx_start, idx_end = blanket_idx
    blankets = [x[i, start:end, :, :, :] for i, (start, end) in enumerate(zip(idx_start, idx_end))]
    return torch.stack(blankets, dim=0)


def compute_blanket_indices(
    indices: torch.Tensor, k: int, trajectory_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Given a set of random indexes, determines the blanket position around it.

    Arguments:
        indices: Random indices for the blanket's center (B).
        k: Number of neighbors to consider on each side to define the "blanket".
        trajectory_size: Total length of the trajectory.

    Returns:
        idx_start: Starting indices for the blankets (B).
        idx_end: Ending indices for the blankets (B).
    """
    idx_start = torch.clip(indices - k, min=0)
    idx_end = torch.clip(indices + k + 1, max=trajectory_size)
    pad_start = torch.clip(k - indices, min=0)
    pad_end = torch.clip(indices + k + 1 - trajectory_size, min=0)
    idx_start -= pad_end
    idx_end += pad_start
    return idx_start, idx_end


def preprocess_for_training(
    x: torch.Tensor, time: torch.Tensor, k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Preprocesses input tensors for training.

    Arguments:
        x: Input tensor (B, T, C, H, W).
        time: Time tensor (B, T, 3).
        k: Number of neighbors to consider on each side to define the "blanket".

    Returns:
        blankets: Tensor (B, 2k+1, C, H, W), representing the extracted blankets of each trajectory.
        time: Tokenized time tensor (B, 3), containing the time of the center state of each blanket.
    """

    # Security
    batch_size, trajectory_size = x.shape[0], x.shape[1]
    assert (
        trajectory_size > 2 * k + 1
    ), f"ERROR () - Trajectory size must be greater than complete blanket ({trajectory_size} > {2 * k + 1})"

    # Preprocessing
    indice_center_blankets = torch.randint(0, trajectory_size, (batch_size,))
    idx_start, idx_end = compute_blanket_indices(
        indices=indice_center_blankets, k=k, trajectory_size=trajectory_size
    )
    x = extract_blankets_in_trajectories(x=x, blanket_idx=(idx_start, idx_end))
    time = [time[b, center] for b, center in enumerate(indice_center_blankets)]
    time = torch.stack(time, dim=0)

    # Flattening input tensor for diffusion and tokenizing time
    return x.flatten(1), time_tokenizer(time)


def training(
    config_dataloader: Dict,
    config_backbone: Dict,
    config_nn: Dict,
    config_training: Dict,
    toy_problem: bool = False,
) -> None:
    r"""Perform training of the Poseidon model.

    Arguments:
        config_dataloader: Dictionary containing dataloader settings (e.g., batch size, shuffle, dataset path).
        config_backbone: Dictionary specifying the backbone architecture settings (e.g., layer sizes, activation functions).
        config_nn: Configuration for the neural network (e.g., optimizer parameters, architecture options).
        config_training: Configuration for the training loop (e.g., number of epochs, learning rate, save frequency).
        toy_problem: Indicates whether or not use the toy dataset.
    """

    # --- Initialization ---
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset based on whether we're using a toy problem or real data
    if toy_problem:
        dl_train, _, _ = get_toy_dataloaders(**config_dataloader)
    else:
        dl_train, _, _ = get_dataloaders(**config_dataloader)

    # Peek at a single batch to determine input dimensions
    data, _ = next(iter(dl_train))
    channels, latitudes, longitudes = data.shape[2:]

    # Initialize backbone, denoiser, and noise scheduler
    backbone = PoseidonBackbone(
        **config_backbone,
        dimensions=(channels, latitudes, longitudes),
        config_nn=config_nn,
        config_region=TOY_DATASET_REGION if toy_problem else DATASET_REGION,
    )
    denoiser = PoseidonDenoiser(backbone)
    noise_scheduler = PoseidonNoiseSchedule()

    backbone.to(device)
    denoiser.to(device)

    # Initialize optimizer
    optimizer = Adam(denoiser.parameters(), lr=config_training["Learning Rate"])

    # --- Setup for saving ---
    nn_name = generate_model_name()
    save_dir = POSEIDON_MODEL / nn_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save configurations and problem information
    save_configurations(
        path=save_dir,
        configs={
            "config_dataloader": config_dataloader,
            "config_backbone": config_backbone,
            "config_nn": config_nn,
            "config_training": config_training,
            "config_problem": {
                "Toy_problem": toy_problem,
                "Channels": channels,
                "Latitudes": latitudes,
                "Longitudes": longitudes,
            },
        },
    )

    # --- Training Loop ---
    # fmt:off
    for epoch in range(1, config_training["Number of Epochs"] + 1):

        for data, time in dl_train:

            # Preprocess the input data
            x, time = preprocess_for_training(data, time, **config_backbone)

            # Compute noise for denoising
            noise = noise_scheduler(size=x.shape[0])

            # Move data, noise, and time to GPU/CPU
            x, time, noise = x.to(device), time.to(device), noise.to(device)

            # Compute loss using the denoiser
            loss = PoseidonLoss(denoiser, x, noise, time)
            if not loss.isnan().any():
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # Display loss and monitor memory usage
            print(f"Info | EPOCH: {epoch} | LOSS: {loss.item()}")

        # Save backbone model at specified intervals
        if epoch % config_training["Save Frequency"] == 0:
            save_model(save_dir, backbone, optimizer, epoch)
            print(f"Model saved for epoch {epoch} at {save_dir}")
