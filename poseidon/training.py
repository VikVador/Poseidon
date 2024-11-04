r"""Poseidon - Tools to perform the training of a denoiser."""

import matplotlib.pyplot as plt
import torch
import torch.optim.lr_scheduler as lr_scheduler
import wandb

from einops import rearrange
from torch.optim import Adam
from tqdm import tqdm
from typing import Dict, Tuple

# isort: split
from poseidon.config import POSEIDON_MODEL
from poseidon.data.const import DATASET_REGION, TOY_DATASET_REGION
from poseidon.data.dataloaders import get_dataloaders, get_toy_dataloaders
from poseidon.diffusion.backbone import PoseidonBackbone
from poseidon.diffusion.denoiser import PoseidonDenoiser
from poseidon.diffusion.loss import PoseidonLoss
from poseidon.diffusion.noise import PoseidonNoiseSchedule
from poseidon.diffusion.sampler import PoseidonSampler
from poseidon.diffusion.tools import (
    compute_blanket_indices,
    extract_blankets_in_trajectories,
    time_tokenizer,
)
from poseidon.network.save import (
    generate_model_name,
    save_backbone,
    save_configuration,
)


def plot_images_from_tensor(tensor, index):
    """
    Given a 4D tensor of shape [6, 33, 128, 128], this function takes an index for the first dimension
    and creates a subplot displaying the 33 images in a grid format.

    Parameters:
        tensor (torch.Tensor): The input tensor of shape [6, 33, 128, 128]
        index (int): The index to slice the first dimension (batch dimension)
    """
    # Select the images for the given index
    images = tensor[:, index]

    # Create the figure with a grid of 33 subplots (you can adjust the grid size as needed)
    fig, axes = plt.subplots(
        6, 6, figsize=(15, 15)
    )  # 6x6 grid for 33 images, adjust figsize as needed

    for i in range(33):
        ax = axes[i // 6, i % 6]
        ax.imshow(images[i].cpu().numpy(), cmap="viridis", vmin=-3, vmax=3)
        ax.axis("off")
    plt.tight_layout()
    return fig


def preprocess_for_training(
    x: torch.Tensor, time: torch.Tensor, k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Helper tool to preprocess data for training.

    Arguments:
        x: Input tensor (B, C, T, H, W).
        time: Time tensor (B, T, 3).
        k: Number of neighbors to consider on each side to define the "blanket".

    Returns:
        blankets: Tensor (B, C, 2k+1, H, W), representing the extracted blankets of each trajectory.
        time: Tokenized time tensor (B, 3), containing the time of the center state of each blanket.
    """
    # Security
    batch_size, _, trajectory_size, _, _ = x.shape
    assert (
        trajectory_size > 2 * k + 1
    ), f"ERROR () - Trajectory size must be greater than complete blanket ({trajectory_size} > {2 * k + 1})"

    # Preprocessing
    indice_center_blankets = torch.randint(0, trajectory_size, (batch_size,))
    idx_start, idx_end, _ = compute_blanket_indices(
        indices=indice_center_blankets, k=k, trajectory_size=trajectory_size
    )
    x = extract_blankets_in_trajectories(x=x, blanket_idx=(idx_start, idx_end))
    x = rearrange(x, "B ... -> B (...)")
    time = [time[b, idx] for b, idx in enumerate(idx_start)]
    time = torch.stack(time, dim=0)
    time = time_tokenizer(time)
    return x, time


def training(
    config_dataloader: Dict,
    config_backbone: Dict,
    config_nn: Dict,
    config_training: Dict,
    wandb_mode: str,
    toy_problem: bool = False,
) -> None:
    r"""Performs the training of a denoiser.

    Arguments:
        config_dataloader: Dataloader settings (e.g., batch size, shuffle, parallel, ...).
        config_backbone: Backbone architecture settings (e.g., blanket size, ...).
        config_nn: Neural network (e.g., architecture options, ...).
        config_training: Configuration for the training loop (e.g., number of epochs, learning rate, save frequency).
        wandb_mode: Mode for wandb (e.g., 'online', 'offline').
        toy_problem: Indicates whether or not use the toy dataset.
    """

    # --- Initialization ---
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if toy_problem:
        dl_train, _, _ = get_toy_dataloaders(**config_dataloader)
    else:
        dl_train, _, _ = get_dataloaders(**config_dataloader)

    # Peek at a single batch to determine input dimensions
    data, _ = next(iter(dl_train))
    _, channels, trajectory_size, latitudes, longitudes = data.shape
    nb_epochs = config_training["Number of Epochs"]
    backbone = PoseidonBackbone(
        **config_backbone,
        dimensions=(channels, trajectory_size, latitudes, longitudes),
        config_nn=config_nn,
        config_region=TOY_DATASET_REGION if toy_problem else DATASET_REGION,
    )
    denoiser = PoseidonDenoiser(backbone)
    noise_scheduler = PoseidonNoiseSchedule()
    backbone.to(device)
    denoiser.to(device)
    optimizer = Adam(denoiser.parameters(), lr=config_training["Learning Rate (Start)"])
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=nb_epochs, eta_min=config_training["Learning Rate (End)"]
    )

    # --- Saving ---
    nn_name = generate_model_name(length=8)
    nn_save_dir = POSEIDON_MODEL / nn_name
    nn_save_dir.mkdir(parents=True, exist_ok=True)
    configs = {
        "config_dataloader": config_dataloader,
        "config_backbone": config_backbone,
        "config_nn": config_nn,
        "config_training": config_training,
        "config_problem": {
            "Name": nn_name,
            "Toy_problem": toy_problem,
            "Channels": channels,
            "Trajectory Size": trajectory_size,
            "Latitudes": latitudes,
            "Longitudes": longitudes,
        },
    }
    save_configuration(path=nn_save_dir, configs=configs)
    wandb.init(
        project="Poseidon-Training-V2",
        mode=wandb_mode,
        name=nn_name,
        config={k: v for d in configs for k, v in configs[d].items()},
    )

    # --- Training Loop ---
    # fmt:off
    for epoch in range(1, nb_epochs + 1):

        averaged_loss = []
        pbar = tqdm(dl_train, desc=f"EPOCH {epoch}/{nb_epochs}")

        for _, (data, time) in enumerate(pbar):

            # Preprocessing data: blanket extraction, flattening, and time tokenization
            x, time = preprocess_for_training(data, time, **config_backbone)

            # Generate noise for denoising process
            noise = noise_scheduler(size=x.shape[0])

            # Move data to the device (GPU/CPU)
            x, time, noise = x.to(device), time.to(device), noise.to(device)

            # Compute the loss and backpropagate
            loss = PoseidonLoss(denoiser, x, noise, time)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Accumulate loss and update progress bar
            averaged_loss.append(loss.item())
            pbar.set_postfix({"| LOSS": loss.item()})

        # --- Checkpointing & Logging ---
        if epoch % config_training["Save Frequency"] == 0:
            save_backbone(nn_save_dir, backbone, optimizer, epoch)

        scheduler.step()
        averaged_loss = torch.tensor(averaged_loss).detach()

        wandb.log({
            "Training/Loss (Averaged Over Batch)": torch.mean(averaged_loss),
            "Training/Learning Rate": scheduler.get_last_lr()[0],
            "Training/Epoch": epoch,
            "Training/Status [%]": 100 * epoch / nb_epochs
        })

        # --- Sampling ---
        if epoch % 10 == 0:

            # Initialization of sampler
            sampler = PoseidonSampler(denoiser=denoiser, steps=256)

            # Dates for evaluation
            dates = ["2019-07-01", "2019-08-01", "2019-09-01"]
            for d in dates:

                # Generates a trajectory
                x = sampler(trajectory_size=7, date=d)

                # Displaying a full trajectory
                for i in [0, 6]:
                    wandb.log({f"Trajectory - {d} | Day {i + 1}": [wandb.Image(plot_images_from_tensor(x, i))]})
                    plt.close()

    wandb.finish()
