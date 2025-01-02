r"""Training."""

import gc
import torch
import wandb

from torch.amp.grad_scaler import GradScaler
from tqdm import tqdm
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
    config_wandb: Dict,
) -> None:
    r"""Launch the training of a `PoseidonDenoiser`.

    Arguments:
        config_problem: Configuration for the problem.
        config_dataloader: Configuration for the dataloaders.
        config_training: Configuration for the training.
        config_unet: Configuration for the UNet (denoiser).
        config_siren: Configuration for the Siren network (spatial embedding).
        config_wandb: Configuration for Weights & Biases.
    """

    wandb.init(
        **config_wandb,
        config={
            "Problem": config_problem,
            "Dataloader": config_dataloader,
            "Training": config_training,
            "UNet": config_unet,
            "Siren": config_siren,
        },
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataloader_train, _, _ = (
        get_toy_dataloaders(**config_dataloader)
        if config_problem["toy_problem"]
        else get_dataloaders(**config_dataloader)
    )

    dataloader_iter = iter(dataloader_train)

    # Extracting dimensions and parameters
    (
        (B, C, _, H, W),
        blanket_neighbors,
        blanket_size,
        steps_training,
        steps_gradient_accumulation,
    ) = (
        next(iter(dataloader_train))[0].shape,
        config_training["blanket_neighbors"],  # Neighbors on each side
        config_training["blanket_neighbors"] * 2 + 1,  # Complete blanket dimension
        config_training["steps_training"],  # One-step is one day
        config_training["steps_gradient_accumulation"],  # Number of steps before optimizer step
    )

    # Setting up building blocks
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
        lr_lambda=lambda t: max(0, 1 - (t / steps_training)),
    )

    poseidon_scheduler_noise = PoseidonNoiseSchedule()

    # Used to handle mixed precision training with gradient accumulation
    scaler = GradScaler()

    #
    # --- Training Loop ---
    #
    # Progression bar
    progress_bar = tqdm(total=steps_training, desc="Training", unit="step")

    # Running average for loss
    loss_average = 0.0

    for step in range(0, steps_training):
        #
        # --- Fetching & Preprocessing Data ---
        #
        try:
            data, _ = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader_train)
            data, _ = next(dataloader_iter)

        # Extractig random blanket from the data and generating noise
        x, noise = (
            preprocessing_for_diffusion(data, blanket_neighbors),
            poseidon_scheduler_noise(batch_size=data.shape[0]),
        )

        # Adding noise to the data
        x_t = x + noise * torch.randn_like(x)

        # Moving data to device
        x_t, noise = x_t.to(device), noise.to(device)

        #
        # --- Forward Pass (Mixed Precision) ---
        #
        with torch.autocast(device_type="cuda"):
            # Denoising the data
            x_t_denoised = poseidon_denoiser(x_t, noise)

            # Computing loss
            loss = PoseidonLoss(x_t, x_t_denoised, noise)

        #
        # --- Gradient Accumulation ---
        #
        # Rescaling the loss
        loss /= steps_gradient_accumulation
        loss_average += loss.item()

        # Backward pass (loss scaled to avoid underflow)
        scaler.scale(loss).backward()

        if step % steps_gradient_accumulation == 0 and step != 0:
            # Optimizer step
            scaler.step(poseidon_optimizer)
            scaler.update()
            poseidon_optimizer.zero_grad()

            # Log the running loss average
            progress_bar.set_postfix({"Loss (AOS)": f"{loss_average:.6f}"})
            wandb.log({
                "Training/Loss (Averaged over Steps)": loss_average,
                "Training/Learning Rate": poseidon_optimizer.param_groups[0]["lr"],
                "Training/Step": step,
                "Training/Completed": step / steps_training,
            })

            # Reseting running average
            loss_average = 0.0

        # Updating learning rate
        if step > steps_gradient_accumulation:
            poseidon_sheduler_lr.step()

        # Updating progress bar
        progress_bar.update(1)

        #
        # --- Cleanup ---
        #
        # Ensure no cuda:CPU memory leaks
        del data, _, x, x_t, noise, x_t_denoised, loss
        torch.cuda.empty_cache()
        gc.collect()

    # End of Training
    progress_bar.close()

    # Closing wandb
    wandb.finish()
