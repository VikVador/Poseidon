r"""Training."""

import dask
import gc
import torch
import wandb

from einops import rearrange
from torch.amp.grad_scaler import GradScaler
from tqdm import tqdm
from typing import Dict

# isort: split
from poseidon.config import PATH_MODEL
from poseidon.data.const import DATASET_VARIABLES
from poseidon.data.dataloaders import get_dataloaders
from poseidon.diffusion.backbone import PoseidonBackbone
from poseidon.diffusion.denoiser import PoseidonDenoiser
from poseidon.diffusion.loss import PoseidonLoss
from poseidon.diffusion.schedulers import PoseidonNoiseScheduler, PoseidonTimeScheduler
from poseidon.tools import wandb_get_hyperparameter_score
from poseidon.training.load import load_backbone
from poseidon.training.optimizer import get_optimizer, safe_gd_step
from poseidon.training.save import PoseidonSave
from poseidon.training.scheduler import get_scheduler

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
    config_nn: Dict,
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
        config_nn: Configuration of the neural network.
        config_wandb: Configuration for Weights & Biases.
        config_cluster: Configuration of the cluster.
    """

    # Avoid deadlocks between training and validation
    dask.config.set(scheduler="synchronous")

    wandb.init(
        **config_wandb,
        config={
            "Problem": config_problem,
            "Dataloader": config_dataloader,
            "Training": config_training,
            "Optimizer": config_optimizer,
            "Scheduler": config_scheduler,
            "Neural Network": config_nn,
            "Cluster": config_cluster,
            "Scores": wandb_get_hyperparameter_score([
                config_dataloader,
                config_training,
                config_optimizer,
                config_nn,
            ]),
        },
    )

    (
        blanket_size,
        sigma_min,
        sigma_max,
        steps_training,
        steps_validation,
        steps_gradient_accumulation,
        steps_logging,
        model_saving,
        model_checkpoint_name,
        model_checkpoint_version,
    ) = (
        config_training["blanket_size"],
        config_training["sigma_min"],
        config_training["sigma_max"],
        config_training["steps_training"],
        config_training["steps_validation"],
        config_training["steps_gradient_accumulation"],
        config_training["steps_logging"],
        config_problem["model_saving"],
        config_problem["model_checkpoint_name"],
        config_problem["model_checkpoint_version"],
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

    # =========================================================
    #                     INITIALIZATION
    # =========================================================
    dataloader_training, dataloader_validation, _ = get_dataloaders(
            trajectory_size = blanket_size,
            **config_dataloader,
            **config_dataloader_additional,
        )

    (B, C, _, X, Y) = next(dataloader_training)[0].shape

    poseidon_backbone = (
        PoseidonBackbone(
            config_nn=config_nn,
            dimensions=(B, C, blanket_size, X, Y),
        )
        if model_checkpoint_name is None
        else load_backbone(
            name_model= model_checkpoint_name,
            best= True if model_checkpoint_version == "best" else False,
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

    if 1 < torch.cuda.device_count():
        poseidon_denoiser = torch.nn.DataParallel(
            poseidon_denoiser,
            device_ids=DEVICE_LIST,
        ).to(DEVICE)

    poseidon_save = PoseidonSave(
        path=PATH_MODEL,
        name_model=wandb.run.name,
        variables=DATASET_VARIABLES,
        dimensions=(B, C, blanket_size, X, Y),
        config_nn=config_nn,
        config_problem=config_problem,
        saving=model_saving,
    )

    optimizer = get_optimizer(
        nn_parameters=poseidon_denoiser.parameters(),
        config_optimizer=config_optimizer,
    )

    scheduler_lr, scheduler_time, scheduler_noise, loss_function = (
        get_scheduler(
            optimizer=optimizer,
            total_steps=int(steps_training / steps_gradient_accumulation),
            config_scheduler=config_scheduler,
        ),
        PoseidonTimeScheduler(),
        PoseidonNoiseScheduler(
            sigma_min=sigma_min,
            sigma_max=sigma_max
        ),
        PoseidonLoss(
            blanket_size=blanket_size,
        ),
    )

    scaler = GradScaler(device=DEVICE)

    loss_aoas, progress_bar = (
        0,
        tqdm(
            total=int(steps_training / steps_logging),
            desc="| POSEIDON | Training",
            unit=f" {steps_logging} step(s)",
        ),
    )

    # =========================================================
    #                        TRAINING
    # =========================================================
    for step, (sample, time) in enumerate(dataloader_training):

        # Preprocessing
        x_0 = rearrange(sample, "B ... -> B (...)")

        # Generating noise levels
        sigma_t = scheduler_noise(
            t = scheduler_time(batch_size = x_0.shape[0])
        )

        # Generating noisy states
        x_t = x_0 + sigma_t * torch.randn_like(x_0)

        # Pushing to device
        x_0, x_t, sigma_t, time = x_0.to(DEVICE), x_t.to(DEVICE), sigma_t.to(DEVICE), time.to(DEVICE)

        # Estimating clean trajectories and measuring error
        x_0_denoised = poseidon_denoiser(x_t = x_t, sigma_t = sigma_t, cond = time)

        loss = loss_function(
            x_0 = x_0,
            x_0_denoised = x_0_denoised,
            sigma_t = sigma_t,
        )

        # Gradients accumulation
        loss       = loss / steps_gradient_accumulation
        loss_aoas += loss.item()
        scaler.scale(loss).backward()

        # ===========================================================================
        #                      LOGGING & OPTIMIZATION & VALIDATION
        # ===========================================================================
        if (step % steps_logging == 0):

            wandb.log({
                "Training/Loss": loss_aoas * steps_gradient_accumulation if step == 0 else loss_aoas,
                "Training/Learning Rate [-]": optimizer.param_groups[0]["lr"],
                "Training/Step [-]": (step + 1),
                "Training/Samples Seen [-]": B * (step + 1),
                "Training/Completed [%]": (step / (steps_training - 2)) * 100,
            })

            progress_bar.set_postfix({"Loss (AoAS) ": f"{(loss_aoas):.4f}"})
            progress_bar.update(1)

            poseidon_save.save(
                loss = loss_aoas,
                optimizer = optimizer,
                scheduler = scheduler_lr,
                model = poseidon_denoiser.module.backbone if torch.cuda.device_count() > 1
                else poseidon_denoiser.backbone,
            )

        if 0 < step:

            if (step % steps_gradient_accumulation == 0):

                safe_gd_step(optimizer=optimizer, grad_clip=1, scaler=scaler)
                scheduler_lr.step()
                loss_aoas = 0.0

                del x_0, x_0_denoised, x_t, sigma_t, time, loss
                torch.cuda.empty_cache()
                gc.collect()

            if (step % steps_validation == 0):
                with torch.no_grad():
                    v_loss, v_count = 0.0, 0

                    for _, (sample, time) in enumerate(dataloader_validation):

                        # Preprocessing
                        x_0 = rearrange(sample, "B ... -> B (...)")

                        # Generating noise levels
                        sigma_t = scheduler_noise(
                            t = scheduler_time(batch_size = x_0.shape[0])
                        )

                        # Generating noisy states
                        x_t = x_0 + sigma_t * torch.randn_like(x_0)

                        # Pushing to device
                        x_0, x_t, sigma_t, time = x_0.to(DEVICE), x_t.to(DEVICE), sigma_t.to(DEVICE), time.to(DEVICE)

                        # Estimating clean trajectories and measuring error
                        v_loss += loss_function(
                            x_0 = x_0,
                            x_0_denoised = poseidon_denoiser(x_t = x_t, sigma_t = sigma_t, cond = time),
                            sigma_t = sigma_t,
                        ).item()

                        # Counting the number of samples
                        v_count += 1

                    wandb.log({"Validation/Loss": v_loss / v_count})
                    del x_0, x_t, sigma_t, time

            # Emergency break
            if steps_training <= step:
                break

    progress_bar.update(1)
    wandb.finish()
