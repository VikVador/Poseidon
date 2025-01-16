r"""Training."""

import gc
import torch
import wandb

from tqdm import tqdm
from typing import Dict

# isort: split
from poseidon.config import PATH_MODEL
from poseidon.data.const import DATASET_REGION, TOY_DATASET_REGION
from poseidon.data.dataloaders import get_dataloaders, get_toy_dataloaders
from poseidon.diffusion.backbone import PoseidonBackbone
from poseidon.diffusion.denoiser import PoseidonDenoiser
from poseidon.diffusion.loss import PoseidonLoss
from poseidon.diffusion.noise import PoseidonNoiseSchedule
from poseidon.training.optimizer import get_optimizer
from poseidon.training.save import PoseidonSave
from poseidon.training.scheduler import get_scheduler
from poseidon.training.tools import preprocessing_for_diffusion

# fmt: off
#
# Constants
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_LIST = [i for i in range(torch.cuda.device_count())]


def training(
    config_problem: Dict,
    config_dataloader: Dict,
    config_training: Dict,
    config_optimizer: Dict,
    config_scheduler: Dict,
    config_unet: Dict,
    config_siren: Dict,
    config_wandb: Dict,
    config_cluster: Dict,
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
        config_cluster: Configuration of the Cluster.
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
            "Cluster": config_cluster,
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
        steps_logging,
        save_model,
        black_sea_region,
    ) = (
        next(iter_dataloader_training)[0].shape,        # Dimension of Black Sea state trajectory
        config_training["blanket_neighbors"],           # Neighbors on each side
        config_training["blanket_neighbors"] * 2 + 1,   # Complete blanket dimension
        config_training["steps_training"],              # One-step is one day
        config_training["steps_gradient_accumulation"], # Number of steps before optimizer step
        config_training["steps_logging"],               # Number of steps before logging
        config_training["save_model"],                  # Whether to save the model or not
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
        ).to(DEVICE),
    )

    wandb.log({
        "Neural Network/Trainable Parameters [-]": sum(
            p.numel() for p in poseidon_denoiser.parameters() if p.requires_grad
        ),
    })

    # Setting up saving tool
    poseidon_save = PoseidonSave(
        path=PATH_MODEL,
        name_model=wandb.run.name,
        dimensions=(B, C, blanket_size, H, W),
        config_unet=config_unet,
        config_siren=config_siren,
        config_problem=config_problem,
        saving=save_model,
    )

    # Launching parallel training
    if torch.cuda.device_count() > 1:
        poseidon_denoiser = torch.nn.DataParallel(
            poseidon_denoiser,
            device_ids=DEVICE_LIST,
        ).to(DEVICE)

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

    # Progression bar showing the accumulated averaged loss
    loss_aoas, progress_bar = (
        0,
        tqdm(
            total=int(steps_training / steps_logging),
            desc="| POSEIDON | Training",
            unit=f" {steps_logging} step(s)",
        )
    )

    for step in range(0, steps_training):
        #
        # TRAINING
        #
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
        loss = loss / steps_gradient_accumulation
        loss.backward()

        # Storing the accumulated loss
        loss_aoas += loss.item()

        #
        # UPDATING & LOGGING
        #
        if step % steps_gradient_accumulation == 0 and step != 0:
            #
            optimizer.step()
            scheduler_lr.step()

            if step % steps_logging == 0 and step != 0:
                #
                progress_bar.set_postfix({
                    "Loss (AoAS) ": f"{(loss_aoas):.4f}",
                })

                wandb.log({
                    "Training/Loss (AoAS)": loss_aoas,
                    "Training/Learning Rate [-]": optimizer.param_groups[0]["lr"],
                    "Training/Step [-]": step,
                    "Training/Samples Seen [-]": B * step,
                    "Training/Completed [%]": (step / steps_training) * 100,
                })

                poseidon_save.save(
                    loss=loss_aoas,
                    model=poseidon_denoiser.backbone,
                    optimizer=optimizer,
                    scheduler=scheduler_lr,
                )

            loss_aoas = 0.0
            optimizer.zero_grad()
            gc.collect()

        if step % steps_logging == 0 and step != 0:
            progress_bar.update(1)

        if step == steps_training - 1:
            poseidon_save.save(
                loss=float("inf"),
                model=poseidon_denoiser.backbone,
                optimizer=optimizer,
                scheduler=scheduler_lr,
            )

        del x, x_noised, noise, loss
        torch.cuda.empty_cache()

    # Finalizing the training
    progress_bar.update(1)
    wandb.finish()
