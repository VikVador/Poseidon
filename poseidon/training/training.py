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
from poseidon.training.optimizer import get_optimizer
from poseidon.training.scheduler import get_scheduler
from poseidon.training.tools import preprocessing_for_diffusion

#
# fmt: off
#
# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def training(
    config_problem: Dict,
    config_dataloader: Dict,
    config_training: Dict,
    config_optimizer: Dict,
    config_scheduler: Dict,
    config_unet: Dict,
    config_siren: Dict,
    config_wandb: Dict,
) -> None:
    r"""Launch the training of a `PoseidonDenoiser`.

    Arguments:
        config_problem: Configuration for the problem.
        config_dataloader: Configuration for the dataloaders.
        config_training: Configuration for the training.
        config_optimizer: Configuration for the optimizer.
        config_scheduler: Configuration for the scheduler.
        config_unet: Configuration for the UNet (denoiser).
        config_siren: Configuration for the Siren network (spatial embedding).
        config_wandb: Configuration for Weights & Biases.
    """

    # Initializing connection to Weights & Biases
    wandb.init(
        **config_wandb,
        config={
            "Problem": config_problem,
            "Dataloader": config_dataloader,
            "Training": config_training,
            "Optimizer": config_optimizer,
            "Scheduler": config_scheduler,
            "UNet": config_unet,
            "Siren": config_siren,
        },
    )

    # Loading dataloaders as infinite iterators
    iter_dataloader_training, _, _ = (
        get_toy_dataloaders(
            **config_dataloader,
            infinite=True,
        )
        if config_problem["toy_problem"]
        else get_dataloaders(
            **config_dataloader,
            infinite=True,
        )
    )

    # Extracting dimensions and parameters
    (
        (B, C, _, H, W),
        blanket_neighbors,
        blanket_size,
        steps_training,
        steps_gradient_accumulation,
        black_sea_region,
    ) = (
        next(iter_dataloader_training)[0].shape,        # Dimension of Black Sea state trajectory
        config_training["blanket_neighbors"],           # Neighbors on each side
        config_training["blanket_neighbors"] * 2 + 1,   # Complete blanket dimension
        config_training["steps_training"],              # One-step is one day
        config_training["steps_gradient_accumulation"], # Number of steps before optimizer step
        TOY_DATASET_REGION                              # Region of interest
        if config_problem["toy_problem"]
        else DATASET_REGION,
    )

    # Setting up denoising network
    poseidon_denoiser = PoseidonDenoiser(
        PoseidonBackbone(
            dimensions=(B, C, blanket_size, H, W),
            config_unet=config_unet,
            config_siren=config_siren,
            config_region=black_sea_region,
            device=DEVICE,
        )
    ).to(DEVICE)

    # Tracking gradients & Number of trainable parameters
    wandb.watch(poseidon_denoiser, log="gradients", log_freq=512)
    wandb.log({
        "Neural Network/Trainable Parameters [-]": sum(
            p.numel() for p in poseidon_denoiser.parameters() if p.requires_grad
        ),
    })

    # Setting up training tools
    optimizer = get_optimizer(
        nn_parameters=poseidon_denoiser.parameters(),
        config_optimizer=config_optimizer,
    )

    scheduler_lr, scheduler_noise = (
        get_scheduler(
            optimizer=optimizer,
            total_steps=int(steps_training / steps_gradient_accumulation),
            config_scheduler=config_scheduler,
        ),
        PoseidonNoiseSchedule(),
    )

    # Handles low losses for gradient accumulation
    scaler = GradScaler()

    # Progression bar showing the accumulated averaged loss
    loss_average, progress_bar = 0, tqdm(total=steps_training, desc="Training", unit="step")

    for step in range(0, steps_training):

        # Fetching data (only state x) and preprocessing it
        x = preprocessing_for_diffusion(
            x=next(iter_dataloader_training)[0],
            k=blanket_neighbors,
        )

        # Generating noise
        noise = scheduler_noise(batch_size=x.shape[0])

        # Noising the clean state
        x_noised = x + noise * torch.randn_like(x)

        # Pushing everything to the device
        x, x_noised, noise = x.to(DEVICE), x_noised.to(DEVICE), noise.to(DEVICE)

        # Denoising the noisy state and computing the loss between clean and denoised states
        loss = PoseidonLoss(
            x=x,
            x_denoised=poseidon_denoiser(x_noised, noise),
            sigma=noise,
        )

        # Computing gradients with scaled loss for gradient accumulation
        scaler.scale(loss / steps_gradient_accumulation).backward()

        # Storing the accumulated loss
        loss_average += loss.item()

        # Gradient accumulation optimizer step
        if step % steps_gradient_accumulation == 0 and step != 0:

            progress_bar.set_postfix({
                "Loss (AoAS) ": f"{(loss_average / steps_gradient_accumulation):.6f}"
            })

            wandb.log({
                "Training/Loss (AoAS)": loss_average / steps_gradient_accumulation,
                "Training/Learning Rate [-]": optimizer.param_groups[0]["lr"],
                "Training/Step [-]": step,
                "Training/Samples Seen [-]": B * step,
                "Training/Completed [%]": (step / steps_training) * 100,
            })

            # Optimizing & Updating
            scaler.step(optimizer)
            scaler.update()
            scheduler_lr.step()

            # Reseting & Cleaning
            loss_average = 0.0
            optimizer.zero_grad()
            gc.collect()

        # Updating & Cleaning
        progress_bar.update(1)
        del x, x_noised, noise, loss
        torch.cuda.empty_cache()

    wandb.finish()
