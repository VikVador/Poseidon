r"""Training."""

import dask
import gc
import torch
import wandb

from torch.amp.grad_scaler import GradScaler
from tqdm import tqdm
from typing import Dict

# isort: split
from poseidon.config import PATH_MODEL
from poseidon.data.const import (
    DATASET_REGION,
    DATASET_VARIABLES,
    TOY_DATASET_REGION,
    TOY_DATASET_VARIABLES,
)
from poseidon.data.dataloaders import get_dataloaders, get_toy_dataloaders
from poseidon.diagnostics.plots import visualize
from poseidon.diffusion.backbone import PoseidonBackbone
from poseidon.diffusion.denoiser import PoseidonDenoiser
from poseidon.diffusion.loss import PoseidonLoss
from poseidon.diffusion.schedulers import PoseidonNoiseScheduler, PoseidonTimeScheduler
from poseidon.tools import wandb_get_hyperparameter_score
from poseidon.training.load import load_backbone
from poseidon.training.optimizer import get_optimizer, safe_gd_step
from poseidon.training.save import PoseidonSave
from poseidon.training.scheduler import get_scheduler
from poseidon.training.tools import extract_random_blankets

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
    config_transformer: Dict,
    config_siren: Dict,
    config_wandb: Dict,
    config_cluster: Dict,
) -> None:
    r"""Launch the training of a class:`PoseidonDenoiser`.

    Arguments:
        config_problem: Configuration for the problem.
        config_dataloader: Configuration for the dataloaders.
        config_training: Configuration for the training.
        config_optimizer: Configuration for the optimizer.
        config_scheduler: Configuration for the scheduler.
        config_unet: Configuration for the UNet.
        config_transformer: Configuration for the Transformer.
        config_siren: Configuration for the Siren network.
        config_wandb: Configuration for Weights & Biases.
        config_cluster: Configuration of the Cluster.
    """

    # Avoid deadlocks between training and validation
    dask.config.set(scheduler="synchronous")

    # Initialize Weights & Biases
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
            "Scores": wandb_get_hyperparameter_score([
                config_dataloader,
                config_training,
                config_optimizer,
                config_unet,
            ]),
        },
    )

    # Unpacking configurations
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
        wandb_mode,
        black_sea_variables,
        black_sea_region,
    ) = (
        config_training["blanket_neighbors"],
        config_training["blanket_neighbors"] * 2 + 1,
        config_training["steps_training"],
        config_training["steps_validation"],
        config_training["steps_gradient_accumulation"],
        config_training["steps_logging"],
        config_problem["model_saving"],
        config_problem["model_checkpoint_name"],
        config_problem["model_checkpoint_version"],
        config_wandb["mode"],
        TOY_DATASET_VARIABLES if config_problem["toy_problem"] else DATASET_VARIABLES,
        TOY_DATASET_REGION    if config_problem["toy_problem"] else DATASET_REGION,
    )

    config_dataloader_additional = {
        "infinite": [True, False, False],
        "steps":    [steps_training, None, None],
        "linspace": [False, True, True],
        "linspace_samples": [
            None,
            3 * 12,
            2 * 12,
        ],
    }

    # Initializing dataloaders
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

    # Dimensions of the state
    (B, C, T, X, Y) = next(dataloader_training)[0].shape

    # Initializing a Backbone
    poseidon_backbone = (
        PoseidonBackbone(
            dimensions=(B, C, blanket_size, X, Y),
            variables=black_sea_variables,
            config_unet=config_unet,
            config_transformer=config_transformer,
            config_siren=config_siren,
            config_region=black_sea_region,
        )
        if model_checkpoint_name is None
        else load_backbone(
            name_model= model_checkpoint_name,
            best= True if model_checkpoint_version == "best" else False,
        )
    )

    # Initializing a Denoiser & Logging number of trainable parameters
    poseidon_denoiser = PoseidonDenoiser(
        backbone=poseidon_backbone.to(DEVICE),
    )

    wandb.log({
        "Neural Network/Trainable Parameters [-]": sum(
            p.numel() for p in poseidon_denoiser.parameters() if p.requires_grad
        ),
    })

    # Initializing multi-GPU support
    if 1 < torch.cuda.device_count():
        poseidon_denoiser = torch.nn.DataParallel(
            poseidon_denoiser,
            device_ids=DEVICE_LIST,
        ).to(DEVICE)

    # Initializing the saving tool
    poseidon_save = PoseidonSave(
        path=PATH_MODEL,
        name_model=wandb.run.name,
        dimensions=(B, C, blanket_size, X, Y),
        variables=black_sea_variables,
        config_unet=config_unet,
        config_siren=config_siren,
        config_problem=config_problem,
        saving=model_saving,
    )

    # Initializing the optimizer
    optimizer = get_optimizer(
        nn_parameters=poseidon_denoiser.parameters(),
        config_optimizer=config_optimizer,
    )

    # Initializing the schedulers and loss function
    scheduler_lr, scheduler_time, scheduler_noise, loss_function = (
        get_scheduler(
            optimizer=optimizer,
            total_steps=int(steps_training / steps_gradient_accumulation),
            config_scheduler=config_scheduler,
        ),
        PoseidonTimeScheduler(),
        PoseidonNoiseScheduler(),
        PoseidonLoss(
            variables=black_sea_variables,
            region=black_sea_region,
            blanket_size=blanket_size,
            use_mask=True,
        ),
    )

    # initializing the gradient scaler
    scaler = GradScaler(device=DEVICE)

    # Initializing tools to track the training
    loss_aoas, progress_bar = (
        0,
        tqdm(
            total=int(steps_training / steps_logging),
            desc="| POSEIDON | Training",
            unit=f" {steps_logging} step(s)",
        ),
    )

    # =========================================================
    #                       TRAINING
    # =========================================================
    for step, (sample, _) in enumerate(dataloader_training):

        # From trajectories, extracting random blankets
        x_0 = extract_random_blankets(x = sample, k = blanket_neighbors)

        # Generating noise levels
        sigma_t = scheduler_noise(
            t = scheduler_time(batch_size = x_0.shape[0])
        )

        # Generating noisy states
        x_t = x_0 + sigma_t * torch.randn_like(x_0)

        # Pushing to device
        x_0, x_t, sigma_t = x_0.to(DEVICE), x_t.to(DEVICE), sigma_t.to(DEVICE)

        # Estimating clean trajectories and measuring error
        x_0_denoised = poseidon_denoiser(x_t = x_t, sigma_t = sigma_t)

        loss = loss_function(
            x_0 = x_0,
            x_0_denoised = x_0_denoised,
            sigma_t = sigma_t,
        )

        # Gradients accumulation
        loss       = loss / steps_gradient_accumulation
        loss_aoas += loss.item()
        scaler.scale(loss).backward()

        # =========================================================================
        #                                 LOGGING
        # =========================================================================
        if (step % steps_logging == 0) or (step == steps_training - 2):

            # Weights & Biases
            wandb.log({
                "Training/Loss (AoAS)": loss_aoas * steps_gradient_accumulation if step == 0 else loss_aoas,
                "Training/Learning Rate [-]": optimizer.param_groups[0]["lr"],
                "Training/Step [-]": (step + 1),
                "Training/Samples Seen [-]": B * (step + 1),
                "Training/Completed [%]": (step / (steps_training - 2)) * 100,
            })

            # Terminal Progression Bar
            progress_bar.set_postfix({"Loss (AoAS) ": f"{(loss_aoas):.4f}"})
            progress_bar.update(1)

            # Saving Model
            poseidon_save.save(
                loss = loss_aoas,
                optimizer = optimizer,
                scheduler = scheduler_lr,
                model = poseidon_denoiser.module.backbone if torch.cuda.device_count() > 1
                else poseidon_denoiser.backbone,
            )

        # =================================================================
        #                            VALIDATION
        # =================================================================
        if (step  % steps_validation == 0) or (step == steps_training - 2):

            with torch.no_grad():

                # ===========================================
                #                    LOSS
                # ===========================================
                # Stores the error made on the validation set
                v_loss = 0.0

                for _, (v_sample, _) in enumerate(dataloader_validation):

                    # From trajectories, extracting random blankets
                    v_x_0 = extract_random_blankets(x = v_sample, k = blanket_neighbors)

                    # Generating noise levels
                    v_sigma_t = scheduler_noise(
                        t = scheduler_time(batch_size = v_x_0.shape[0])
                    )

                    # Generating noisy states
                    v_x_t = v_x_0 + v_sigma_t * torch.randn_like(v_x_0)

                    # Pushing to device
                    v_x_0, v_x_t, v_sigma_t = v_x_0.to(DEVICE), v_x_t.to(DEVICE), v_sigma_t.to(DEVICE)

                    # Estimating clean trajectories and measuring error
                    v_loss += loss_function(
                        x_0 = v_x_0,
                        x_0_denoised = poseidon_denoiser(x_t = v_x_t, sigma_t = v_sigma_t),
                        sigma_t = v_sigma_t,
                    ).item()

                # Weights & Biases
                wandb.log({"Validation/Loss (Averaged)": v_loss / config_dataloader_additional["linspace_samples"][1]})

                # ===========================================
                #                VISUALIZATION
                # ===========================================
                if wandb_mode == "online":

                    visualize(
                        wandb_mode=wandb_mode,
                        variables=black_sea_variables,
                        region=black_sea_region,
                        dimensions=(C, X, Y),
                        denoiser=poseidon_denoiser.module
                        if torch.cuda.device_count() > 1
                        else poseidon_denoiser,
                    )

        # ===========================================================================
        #                             OPTIMIZATION STEP
        # ===========================================================================
        if 0 < step:
            if (step % steps_gradient_accumulation == 0) or (step == steps_training - 2):

                safe_gd_step(optimizer=optimizer, grad_clip=1, scaler=scaler)
                scheduler_lr.step()
                loss_aoas = 0.0
                gc.collect()

        # Cleaning
        del x_0, x_t, sigma_t, loss
        torch.cuda.empty_cache()

        # Emergency stop
        if steps_training <= step:
            break

    # Finalizing the training
    progress_bar.update(1)
    wandb.finish()
