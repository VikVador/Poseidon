r"""Training."""

import dask
import gc
import matplotlib.pyplot as plt
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
from poseidon.diffusion.sampler import PoseidonEulerSampler
from poseidon.training.load import load_backbone
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

    # Avoid deadlocks between training and validation
    dask.config.set(scheduler="synchronous")

    (
        blanket_neighbors,
        blanket_size,
        steps_training,
        steps_validation,
        steps_gradient_accumulation,
        steps_logging,
        model_saving,
        model_checkpoint_name,
        model_checkpoint_version,
        black_sea_region,
    ) = (
        config_training["blanket_neighbors"],           # Neighbors on each side
        config_training["blanket_neighbors"] * 2 + 1,   # Complete blanket dimension
        config_training["steps_training"],              # One-step is one day
        config_training["steps_validation"],            # Number of steps before validation
        config_training["steps_gradient_accumulation"], # Number of steps before optimizer step
        config_training["steps_logging"],               # Number of steps before logging
        config_problem["model_saving"],                 # Whether to save the model or not
        config_problem["model_checkpoint_name"],        # Name of the model checkpoint (if any)
        config_problem["model_checkpoint_version"],     # Version of the model checkpoint (best or last)
        TOY_DATASET_REGION                              # Region of interest
        if config_problem["toy_problem"]
        else DATASET_REGION,
    )

    # Loading dataloaders
    config_dataloader_additional = {
        "infinite": [True, False, False],      # Infinite iterator configuration
        "steps": [steps_training, None, None], # Maximum number of steps before stopping training
        "linspace": [False, True, True],       # Linear temporal subsampling for validation and testing
        "linspace_samples": [                  # Number of samples for each subsampling, ~1 sample/week for robustness
            None,
            3 * 12,
            2 * 12,
        ],
    }

    dataloader_training, dataloader_validation, _ = (
        get_toy_dataloaders(
            **config_dataloader,
            **config_dataloader_additional,
        )
        if config_problem["toy_problem"]
        else get_dataloaders(
            **config_dataloader,
            **config_dataloader_additional,
        )
    )

    # Dimension of Black Sea state trajectory
    (B, C, _, H, W) = next(dataloader_training)[0].shape

    # Setting up denoising network from scratch or loading from checkpoint
    poseidon_backbone = (
        PoseidonBackbone(
            dimensions=(B, C, blanket_size, H, W),
            config_unet=config_unet,
            config_siren=config_siren,
            config_region=black_sea_region,
        )
        if model_checkpoint_name is None
        else load_backbone(
            name_model=model_checkpoint_name,
            best=True if model_checkpoint_version == "best" else False,
        )
    )

    poseidon_denoiser = PoseidonDenoiser(
        backbone=poseidon_backbone.to(DEVICE),
    )

    wandb.log({
        "Neural Network/Trainable Parameters [-]": sum(
            p.numel() for p in poseidon_denoiser.parameters() if p.requires_grad
        ),
    })

    # Placing the model on multiple GPUs
    if torch.cuda.device_count() > 1:
        poseidon_denoiser = torch.nn.DataParallel(
            poseidon_denoiser,
            device_ids=DEVICE_LIST,
        ).to(DEVICE)

    # Setting up saving tool
    poseidon_save = PoseidonSave(
        path=PATH_MODEL,
        name_model=wandb.run.name,
        dimensions=(B, C, blanket_size, H, W),
        config_unet=config_unet,
        config_siren=config_siren,
        config_problem=config_problem,
        saving=model_saving,
    )

    # Setting up training tool
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

    loss_aoas, progress_bar = (
        0,
        tqdm(
            total=int(steps_training / steps_logging),
            desc="| POSEIDON | Training",
            unit=f" {steps_logging} step(s)",
        ),
    )

    # ========================
    # ======= TRAINING =======
    # ========================
    for step, (x, _) in enumerate(dataloader_training):
        #
        x = preprocessing_for_diffusion(
            x=x,
            k=blanket_neighbors,
        )

        # Generating noise
        noise = scheduler_noise(batch_size=x.shape[0])

        # Noising the clean state
        x_noised = x + noise * torch.randn_like(x)

        # Pushing everything to the device
        x, x_noised, noise = (
            x.to(DEVICE),
            x_noised.to(DEVICE),
            noise.to(DEVICE),
        )

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

        # ========================
        # ======== LOGGING =======
        # ========================
        if ((step + 1) % steps_logging == 0 and step != 0) or (step == steps_training - 2):
            #
            progress_bar.set_postfix({
                "Loss (AoAS) ": f"{(loss_aoas):.4f}",
            })
            progress_bar.update(1)

            wandb.log({
                "Training/Loss (AoAS)": loss_aoas,
                "Training/Learning Rate [-]": optimizer.param_groups[0]["lr"],
                "Training/Step [-]": (step + 1),
                "Training/Samples Seen [-]": B * (step + 1),
                "Training/Completed [%]": (step / (steps_training - 2)) * 100,
            })

            poseidon_save.save(
                loss=loss_aoas,
                optimizer=optimizer,
                scheduler=scheduler_lr,
                model=poseidon_denoiser.module.backbone
                if torch.cuda.device_count() > 1
                else poseidon_denoiser.backbone,
            )

        # ========================
        # ====== VALIDATING ======
        # ========================
        if ((step + 1) % steps_validation == 0 and step != 0) or (step == steps_training - 2):
            with torch.no_grad():

                # === Average Validation Loss ===
                #
                loss_avg_val = 0.0

                for x_val, _ in dataloader_validation:
                    #
                    x_val = preprocessing_for_diffusion(
                        x=x_val,
                        k=blanket_neighbors,
                    )

                    # Generating noise
                    noise_val = scheduler_noise(batch_size=x_val.shape[0])

                    # Noising the clean state
                    x_noised_val = x_val + noise_val * torch.randn_like(x_val)

                    # Pushing everything to the device
                    x_val, x_noised_val, noise_val = (
                        x_val.to(DEVICE),
                        x_noised_val.to(DEVICE),
                        noise_val.to(DEVICE),
                    )

                    loss_avg_val += PoseidonLoss(
                        x=x_val,
                        x_denoised=poseidon_denoiser(x_noised_val, noise_val),
                        sigma=noise_val,
                    ).item()

                wandb.log({
                    "Validation/Loss (Averaged)": loss_avg_val
                    / config_dataloader_additional["linspace_samples"][1],
                })

                # === Visualisation ===
                #
                if config_wandb["mode"] == "online":

                    sampler = PoseidonEulerSampler(
                        denoiser=poseidon_denoiser.module
                        if torch.cuda.device_count() > 1
                        else poseidon_denoiser
                        )

                    # Forecasts dimensions
                    fore_size, traj_size, steps = 3, 7, 128

                    # Generating a trajetory
                    euler_trajectory = sampler.forward(
                        trajectory_size=traj_size,
                        forecast_size=fore_size,
                        steps=steps)

                    # Visualizing the trajectory
                    fig, axs = plt.subplots(fore_size, traj_size, figsize=(15, 5))
                    for i in range(fore_size):
                        for j in range(traj_size):
                            axs[i, j].imshow(euler_trajectory[i, 0, j].detach().cpu().numpy(), cmap="viridis")
                            axs[i, j].axis("off")
                    plt.tight_layout()
                    wandb.log({
                        "Validation/Trajectory": wandb.Image(fig),
                    })

        # ========================
        # ======= UPDATING =======
        # ========================
        if ((step + 1) % steps_gradient_accumulation == 0 and step != 0) or (
            step == steps_training - 2
        ):
            optimizer.step()
            scheduler_lr.step()
            optimizer.zero_grad()
            loss_aoas = 0.0
            gc.collect()

        # Cleaning
        del x, x_noised, noise, loss
        torch.cuda.empty_cache()

        # Emergency stop
        if steps_training <= step:
            break

    # Finalizing the training
    progress_bar.update(1)
    wandb.finish()
