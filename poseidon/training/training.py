r"""Training."""

import dask
import os
import torch
import torch.distributed as dist
import wandb

from einops import rearrange
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
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
def setup_distributed():
    r"""Initialize distributed training environment."""

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Running under torchrun
        rank       = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        # Initialize process group (torchrun sets up env variables)
        dist.init_process_group(backend="nccl", init_method="env://")

        # Set device for this process
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        # Set matmul precision for performance on modern GPUs
        torch.set_float32_matmul_precision("high")
        return rank, local_rank, world_size, device, True

    else:
        # Single-GPU or CPU fallback
        rank, local_rank, world_size = 0, 0, 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return rank, local_rank, world_size, device, False


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

    # Initialization of distributed training
    rank, local_rank, world_size, device, is_distributed = setup_distributed()

    # Extracting distributed configuration
    config_distributed_raw = config_training.get("config_distributed", {})

    # Helper to extract value (handles both list format ["nccl"] and direct format "nccl")
    def get_config_value(config, key, default):
        value = config.get(key, default)
        return value[0] if isinstance(value, list) else value

    config_distributed = {
        "backend":                 get_config_value(config_distributed_raw, "backend", "nccl"),
        "find_unused_parameters":  get_config_value(config_distributed_raw, "find_unused_parameters", False),
        "gradient_as_bucket_view": get_config_value(config_distributed_raw, "gradient_as_bucket_view", True),
    }

    # Avoid deadlocks between training and validation
    dask.config.set(scheduler="synchronous")

    # Initialize Weights & Biases
    if rank == 0:
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
                "Distributed": {
                    "world_size": world_size,
                    "backend": config_distributed.get("backend", "nccl"),
                },
                "Scores": wandb_get_hyperparameter_score([
                    config_dataloader,
                    config_training,
                    config_optimizer,
                    config_nn,
                ]),
            },
        )
    else:
        wandb.init(mode="disabled")

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
            3 * 52,
            2 * 52,
        ],
    }

    # =========================================================
    #                     INITIALIZATION
    # =========================================================
    dataloader_training, dataloader_validation, _ = get_dataloaders(
            trajectory_size = blanket_size,
            **config_dataloader,
            **config_dataloader_additional,
            rank=rank,
            world_size=world_size,
            is_distributed=is_distributed,
        )

    # Synchronize all processes before training
    if is_distributed:
        dist.barrier()

    (B, C, _, X, Y) = next(dataloader_training)[0].shape

    # Creation of backbone and denoiser
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
    ).to(device)

    poseidon_denoiser = PoseidonDenoiser(
        backbone=poseidon_backbone,
    )

    # Logging number of trainable parameters
    if rank == 0:
        wandb.log({
            "Neural Network/Trainable Parameters [-]": sum(
                p.numel() for p in poseidon_denoiser.parameters() if p.requires_grad
            ),
        })

    # Parallelization of the model
    if is_distributed:
        poseidon_denoiser = DDP(poseidon_denoiser,
            device_ids              = [local_rank],
            output_device           = local_rank,
            find_unused_parameters  = config_distributed.get("find_unused_parameters", False),
            gradient_as_bucket_view = config_distributed.get("gradient_as_bucket_view", True),
        )

    elif torch.cuda.device_count() > 1:
        poseidon_denoiser = torch.nn.DataParallel(poseidon_denoiser,
            device_ids = [i for i in range(torch.cuda.device_count())],
        ).to(device)

    # Setup of saving utility
    poseidon_save = PoseidonSave(
        path=PATH_MODEL,
        name_model=wandb.run.name,
        variables=DATASET_VARIABLES,
        dimensions=(B, C, blanket_size, X, Y),
        config_nn=config_nn,
        config_problem=config_problem,
        saving=model_saving,
        rank=rank,
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

    scaler = GradScaler(enabled=True)

    # Progression bar
    if rank == 0:
        progress_bar = tqdm(
            total=int(steps_training / steps_logging),
            desc="| POSEIDON | Training",
            unit=f" {steps_logging} step(s)",
        )
    else:
        progress_bar = None

    # Storage for accumulated loss
    loss_aoas = 0

    # =========================================================
    #                        TRAINING
    # =========================================================
    for step, (sample, time) in enumerate(dataloader_training):

        # Preprocessing
        x_0 = rearrange(sample, "B ... -> B (...)")

        # Generating noise levels
        sigma_t = scheduler_noise(t = scheduler_time(batch_size = x_0.shape[0]))

        # Generating noisy states
        x_t = x_0 + sigma_t * torch.randn_like(x_0)

        # Pushing to device
        x_0, x_t, sigma_t, time = x_0.to(device), x_t.to(device), sigma_t.to(device), time.to(device)

        # Mixed precision forward pass
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):

            # Estimating clean trajectories and measuring error
            x_0_denoised = poseidon_denoiser(x_t = x_t, sigma_t = sigma_t, cond = time)

            # Computing loss
            loss = loss_function(x_0 = x_0, x_0_denoised = x_0_denoised, sigma_t = sigma_t,)

        # Gradients accumulation
        loss       = loss / steps_gradient_accumulation
        loss_aoas += loss.item()

        # Only sync gradients on last accumulation step
        is_last_accumulation_step = ((step + 1) % steps_gradient_accumulation == 0)

        if is_distributed and not is_last_accumulation_step:
            with poseidon_denoiser.no_sync():
                scaler.scale(loss).backward()
        else:
            scaler.scale(loss).backward()

        # Cleaning memory
        del x_0, x_0_denoised, x_t, sample, sigma_t, time

        # ===========================================================================
        #                      LOGGING & OPTIMIZATION & VALIDATION
        # ===========================================================================
        if (step % steps_logging == 0):
            if rank == 0:

                # Coputing mean loss over logging interval
                mean_loss = (loss_aoas * steps_gradient_accumulation) / steps_logging

                # Logging
                progress_bar.set_postfix({"Loss (AoAS) ": f"{(mean_loss):.4f}"})
                progress_bar.update(1)
                wandb.log({
                    "Training/Loss": mean_loss,
                    "Training/Learning Rate [-]": optimizer.param_groups[0]["lr"],
                    "Training/Step [-]": (step + 1),
                    "Training/Samples Seen [-]": B * world_size * (step + 1),
                    "Training/Completed [%]": (step / (steps_training - 2)) * 100,
                })

            # Reset accumulator for next logging interval
            loss_aoas = 0.0

            # Unwrap model for saving
            if is_distributed or torch.cuda.device_count() > 1:
                model_to_save = poseidon_denoiser.module.backbone
            else:
                model_to_save = poseidon_denoiser.backbone

            poseidon_save.save(
                loss = loss_aoas,
                optimizer = optimizer,
                scheduler = scheduler_lr,
                model = model_to_save,
            )

        if 0 < step:

            # Optimization step
            if (step % steps_gradient_accumulation == 0):
                safe_gd_step(optimizer=optimizer, grad_clip=1, scaler=scaler)
                scheduler_lr.step()
                loss_aoas = 0.0
                del loss

            # Validation step
            if (step % steps_validation == 0):

                # Synchronize before validation
                if is_distributed:
                    dist.barrier()

                with torch.no_grad():
                    v_loss, v_count = 0.0, 0
                    for _, (sample, time) in enumerate(dataloader_validation):

                        # Preprocessing
                        x_0 = rearrange(sample, "B ... -> B (...)")

                        # Generating noise levels
                        sigma_t = scheduler_noise(t = scheduler_time(batch_size = x_0.shape[0]))

                        # Generating noisy states
                        x_t = x_0 + sigma_t * torch.randn_like(x_0)

                        # Pushing to device
                        x_0, x_t, sigma_t, time = x_0.to(device), x_t.to(device), sigma_t.to(device), time.to(device)

                        # Mixed precision validation
                        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):

                            # Estimating clean trajectories and measuring error
                            v_loss += loss_function(
                                x_0 = x_0,
                                x_0_denoised = poseidon_denoiser(x_t = x_t, sigma_t = sigma_t, cond = time),
                                sigma_t = sigma_t,
                            ).item()

                        # Counting the number of samples
                        v_count += 1

                    # AGGREGATE ACROSS ALL RANKS
                    if is_distributed:

                        v_loss_tensor  = torch.tensor(v_loss, device=device)
                        v_count_tensor = torch.tensor(v_count, device=device)

                        dist.all_reduce(v_loss_tensor, op=dist.ReduceOp.SUM)
                        dist.all_reduce(v_count_tensor, op=dist.ReduceOp.SUM)

                        v_loss  = v_loss_tensor.item()
                        v_count = v_count_tensor.item()

                    if rank == 0:
                        wandb.log({"Validation/Loss": v_loss / v_count})

                    # Cleaning memory
                    del x_0, x_t, sigma_t, time

            # Emergency break
            if steps_training <= step:
                break

    # Cleanup (Weights & Biases and distributed training)
    if rank == 0 and progress_bar is not None:
        progress_bar.update(1)
        progress_bar.close()

    wandb.finish()

    if is_distributed:
        dist.destroy_process_group()
