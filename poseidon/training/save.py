r"""Helper tools to perform saving & loading operations"""

import torch
import yaml

from pathlib import Path
from torch.optim import Optimizer, lr_scheduler
from typing import Dict, Optional

# isort: split
from poseidon.diffusion.backbone import PoseidonBackbone


class PoseidonSave:
    r"""A helper tool to ease the process of saving during training.

    Arguments:
        path: Path to root folder.
        name_model: Name of the model.
        dimensions: Input tensor dimensions (B, C, K, H, W).
        config_unet: Configuration the UNet architecture.
        config_siren: Configuration for the Siren architecture.
        config_problem: Configuration of the problem.
        saving: Whether to save or not.
    """

    def __init__(
        self,
        path: Path,
        name_model: str,
        dimensions: tuple,
        config_unet: dict,
        config_siren: dict,
        config_problem: dict,
        saving: bool = True,
    ):
        super().__init__()

        self.path = path
        self.name_model = name_model
        self.loss_best = float("inf")
        self.saving = saving

        if self.saving:
            #
            # Saving configurations
            list_configs, list_names = (
                [
                    {
                        "dimensions": list(dimensions),
                    },
                    config_unet,
                    config_siren,
                    config_problem,
                ],
                ["dimensions", "unet", "siren", "problem"],
            )

            for config, name in zip(list_configs, list_names):
                save_configuration(
                    path=self.path, config=config, name_model=self.name_model, name_config=name
                )

    def save(
        self,
        loss: float,
        model: PoseidonBackbone,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[lr_scheduler] = None,
    ) -> None:
        r"""Saves model, optimizer & scheduler.

        Information:
            A backup is saved at first, then the true model and tools
            are saved. This allows to recover anything in case the
            training is interrupted abruptly.
        """
        if self.saving:
            #
            # Saving tools and last model with backup protection
            for n in [self.name_model, self.name_model + "/__backup__"]:
                #
                save_tools(
                    path=self.path,
                    name_model=n,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )

                save_backbone(
                    path=self.path,
                    name_model=n,
                    name_state="last",
                    model=model,
                )

            # Saving best model
            if loss < self.loss_best:
                for n in [self.name_model, self.name_model + "/__backup__"]:
                    #
                    save_backbone(
                        path=self.path,
                        name_model=n,
                        name_state="best",
                        model=model,
                    )

                # Updating the best loss
                self.loss_best = loss


def save_backbone(
    path: Path,
    model: PoseidonBackbone,
    name_model: str,
    name_state: str,
) -> None:
    r"""Saves a `PoseidonBackbone` model.

    Arguments:
        path: Path to folder in which save the model.
        model: Backbone to save in its current state.
        name_model: Name of the model.
        name_save: Name of the state.
    """
    saving_folder = path / f"{name_model}" / "models"
    if not saving_folder.exists():
        saving_folder.mkdir(parents=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
        },
        saving_folder / f"{name_state}.pth",
    )


def save_configuration(
    path: Path,
    config: Dict,
    name_model: str,
    name_config: str,
) -> None:
    r"""Saves a configuration as a .yml file.

    Arguments:
        path: Path to folder in which save the configuration.
        configs: Dictionary containing a configuration file.
        name_model: Name of the model.
        name_config: Name of the configuration.
    """
    config_folder = path / f"{name_model}" / "configurations"
    if not config_folder.exists():
        config_folder.mkdir(parents=True)

    with open(config_folder / f"{name_config}.yml", "w") as file:
        yaml.dump(config, file)


def save_tools(
    path: Path,
    name_model: str,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[lr_scheduler] = None,
) -> None:
    r"""Saves the optimizer and scheduler.

    Arguments:
        path: Path to folder in which save the tools.
        optimizer: Optimizer to save.
        scheduler: Scheduler to save.
        name_model: Name of the model trained with these tools.
    """
    tools_folder = path / f"{name_model}" / "tools"
    if not tools_folder.exists():
        tools_folder.mkdir(parents=True)

    if optimizer is not None:
        torch.save(
            {
                "optimizer_state_dict": optimizer.state_dict(),
            },
            tools_folder / "optimizer.pth",
        )

    if scheduler is not None:
        torch.save(
            {
                "scheduler_state_dict": scheduler.state_dict(),
            },
            tools_folder / "scheduler.pth",
        )
