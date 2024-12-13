r"""Training."""

import torch
import wandb

from typing import Dict

# isort: split
from poseidon.data.const import DATASET_REGION, TOY_DATASET_REGION
from poseidon.data.dataloaders import get_dataloaders, get_toy_dataloaders
from poseidon.diffusion.backbone import PoseidonBackbone
from poseidon.diffusion.denoiser import PoseidonDenoiser
from poseidon.diffusion.loss import PoseidonLoss
from poseidon.diffusion.noise import PoseidonNoiseSchedule
from poseidon.training.optimizer import SOAP
from poseidon.training.tools import preprocessing_for_diffusion


def training(
    config_problem: Dict,
    config_dataloader: Dict,
    config_training: Dict,
    config_unet: Dict,
    config_siren: Dict,
    wandb_mode: str,
) -> None:
    r"""Launch the training of a `PoseidonDenoiser`.

    Arguments:
        config_problem: Configuration for the problem.
        config_dataloader: Configuration for the dataloaders.
        config_backbone: Backbone architecture settings (e.g., blanket size, ...).
        config_nn: Neural network (e.g., architecture options, ...).
        config_training: Configuration for the training loop (e.g., number of epochs, learning rate, save frequency).
        wandb_mode: Mode for wandb (e.g., 'online', 'offline').
        toy_problem: Indicates whether or not use the toy dataset.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialization
    wandb.init(project="Poseidon-Training-V3", mode=wandb_mode)

    dataloader_train, dataloader_valid, dataloader_test = (
        get_toy_dataloaders(**config_dataloader)
        if config_problem["toy_problem"]
        else get_dataloaders(**config_dataloader)
    )

    (B, C, _, H, W), blanket_neighbors, blanket_size, number_epochs = (
        next(iter(dataloader_train))[0].shape,
        config_training["blanket_neighbors"],  # Neighbors on each side
        config_training["blanket_neighbors"] * 2 + 1,  # Complete blanket dimension
        config_training["number_epochs"],
    )

    # Setup main training components
    poseidon_denoiser = PoseidonDenoiser(
        PoseidonBackbone(
            dimensions=(B, C, blanket_size, H, W),
            config_unet=config_unet,
            config_siren=config_siren,
            config_region=TOY_DATASET_REGION if config_problem["toy_problem"] else DATASET_REGION,
            device=device,
        )
    ).to(device)

    poseidon_optimizer = SOAP(
        params=poseidon_denoiser.parameters(),
        lr=config_training["learning_rate"],
        weight_decay=config_training["weight_decay"],
    )

    poseidon_sheduler_lr = torch.optim.lr_scheduler.LambdaLR(
        optimizer=poseidon_optimizer,
        lr_lambda=lambda t: max(0, 1 - (t / number_epochs)),  # Linear decay
    )

    poseidon_scheduler_noise = PoseidonNoiseSchedule()

    # --- Training Loop ---
    # fmt:off
    for epoch in range(1, number_epochs + 1):

        averaged_loss = []

        for data, _ in dataloader_train:

            # Preprocessing data: blanket extraction, flattening, and time tokenization
            x = preprocessing_for_diffusion(data, blanket_neighbors)

            # Generate noise for denoising process
            noise = poseidon_scheduler_noise(batch_size=x.shape[0])

            # Generating noisy data
            x_t = x + noise * torch.randn_like(x)

            # Move data to the device (GPU/CPU)
            x_t, noise = x_t.to(device), noise.to(device)

            # Compute the loss and backpropagate
            x_t_denoised = poseidon_denoiser(x_t, noise)

            loss = PoseidonLoss(x_t, x_t_denoised, noise)
            loss.backward()
            poseidon_optimizer.step()
            poseidon_optimizer.zero_grad()

            # Accumulate loss and update progress bar
            averaged_loss.append(loss.item())

            wandb.log({"Training/Loss (Instantenous)": loss.item()})

        # Update learning rate
        poseidon_sheduler_lr.step()

        averaged_loss = torch.tensor(averaged_loss).detach()

        wandb.log({
            "Training/Loss (Averaged Over Batch)": torch.mean(averaged_loss),
            "Training/Epoch": epoch,
            "Training/Status [%]": 100 * epoch / number_epochs,
            "Training/Learning Rate": poseidon_optimizer.param_groups[0]["lr"]
        })

    wandb.finish()
