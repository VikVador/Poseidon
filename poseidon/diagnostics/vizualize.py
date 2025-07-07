r"""Visualisation tools."""

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import wandb
import xarray as xr

from poseidon.data.const import DATASET_REGION, DATASET_VARIABLES, DATASET_VARIABLES_OCEAN
from typing import Dict, Optional

# isort: split
from poseidon.config import LOCAL, PATH_DATA, PATH_MODEL
from poseidon.data.const import (
    DATASET_VARIABLES_SURFACE,
    TOY_DATASET_REGION,
    TOY_DATASET_VARIABLES,
    TOY_DATASET_VARIABLES_OCEAN,
    TOY_DATASET_VARIABLES_SURFACE,
)
from poseidon.data.mappings import from_tensor_to_xarray
from poseidon.diagnostics.const import CMAPS_SURF, TRANSLATION


def plot_unconditional(config: Dict, config_setup: Dict) -> None:
    r"""Visualizes unconditional nowcasts.

    Args:
        config: Model configuration dictionary.
        config_setup: Configuration setup dictionary.
    """

    # Weights and Biases
    wandb.init(
        project=config_setup["wandb_project"],
        mode=config_setup["wandb_mode"],
        name=config["model"],
        resume="allow",
    )

    # Initialization
    toy_problem = config["toy_problem"]
    region = TOY_DATASET_REGION if toy_problem else DATASET_REGION
    variables = TOY_DATASET_VARIABLES if toy_problem else DATASET_VARIABLES
    variables_surf = TOY_DATASET_VARIABLES_SURFACE if toy_problem else DATASET_VARIABLES_SURFACE
    variables_ocean = TOY_DATASET_VARIABLES_OCEAN if toy_problem else DATASET_VARIABLES_OCEAN
    folder = PATH_MODEL / config["model"] / "nowcasts" / "unconditional"

    # Loading ground truth
    x_gt = (
        xr.open_zarr(PATH_DATA)
        .sel(time=slice("2020-01-01", "2020-12-31"))
        .isel(**region, time=np.linspace(0, 365, 12, dtype=int))
        .load()
    )

    # Loading nowcasts
    x = torch.cat(
        [
            torch.load(
                folder / f"nowcast_unconditional_{i}.pt", map_location="cpu", weights_only=True
            )
            for i in range(24)
            if (folder / f"nowcast_unconditional_{i}.pt").exists()
        ],
        dim=0,
    )

    # Convert to xarray
    x = from_tensor_to_xarray(x=x, variables=variables, region=region)

    def plot_variable(v: str, level: Optional[int] = None, saving: bool = False) -> None:
        r"""Helper function to plot a variable."""
        if level is not None:
            x_v = x[v].isel(level=level).values[:, 0]
            x_gt_v = x_gt[v].isel(level=level).values
        else:
            x_v = x[v].values[:, 0]
            x_gt_v = x_gt[v].values

        fig, axs = plt.subplots(6, 6, figsize=(26, 14))

        # Add labels
        fig.text(
            0.11,
            (axs[0, 0].get_position().y0 + axs[1, 0].get_position().y1) / 2,
            "Examples",
            va="center",
            ha="left",
            fontsize=20,
            rotation=90,
        )
        fig.text(
            0.11,
            (axs[3, 0].get_position().y0 + axs[4, 0].get_position().y1) / 2,
            "Samples",
            va="center",
            ha="left",
            fontsize=20,
            rotation=90,
        )

        # Add horizontal separator
        y_sep = (axs[1, 0].get_position().y0 + axs[2, 0].get_position().y1) / 2
        fig.add_artist(
            plt.Line2D(
                [0.135, 0.90],
                [y_sep, y_sep],
                color="black",
                linewidth=2,
                transform=fig.transFigure,
            )
        )

        # Plot data
        data = np.concatenate([x_gt_v, x_v], axis=0)
        for i, ax in enumerate(axs.flat):
            q_min, q_max = np.nanquantile(data[i], [0.1, 0.99])
            ax.imshow(np.flipud(data[i]), cmap=CMAPS_SURF[v], vmin=q_min, vmax=q_max)
            ax.axis("off")
            if i == 0:
                level_str = (
                    f" | Level {level} = {x_gt.level.values[level]:.3f} [m]"
                    if level is not None
                    else ""
                )
                ax.set_title(f"{TRANSLATION[v]}{level_str}", fontsize=20)

        # Sending to Weights and Biases
        wandb.log({
            f"PRIOR | {TRANSLATION[v]} / Level {level}"
            if level is not None
            else f"PRIOR | {TRANSLATION[v]}": wandb.Image(fig)
        })

        # Save figure
        if saving:
            # Path to save the figure
            save_path = f"{LOCAL}/poseidon/metrics/visualizations/{config['model']}/prior/{v}"
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)

            fig.savefig(
                f"{save_path}/level_{l}.png" if level is not None else f"{save_path}/surface.png",
                bbox_inches="tight",
                dpi=350,
            )

        plt.close(fig)

    for v in variables_ocean:
        for l in range(x.level.size):
            plot_variable(v=v, level=l, saving=config_setup["saving"])

    for v in variables_surf:
        plot_variable(v=v, saving=config_setup["saving"])

    wandb.finish()
