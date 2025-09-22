r"""Tools to generate ensembles."""

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import wandb
import xarray as xr

from typing import Dict

# isort: split
from poseidon.config import PATH_MODEL, PATH_POS_LOCAL, PATH_STAT
from poseidon.data.const import DATASET_REGION, DATASET_VARIABLES_OCEAN
from poseidon.data.dataloaders import get_dataloaders
from poseidon.data.mask import generate_trajectory_mask
from poseidon.diagnostics.const import CMAPS_SURF, TRANSLATION


def visualize_ensemble_prior(date: str, config: Dict, config_wandb: Dict) -> None:
    r"""Visualizes an ensemble generated from P(X|d).

    Arguments:
        date: Ensemble date (MM-DD).
        config: Configuration for generation.
        config_wandb: Configuration setup dictionary.
    """

    # Initialization of Weights and Biases
    wandb.init(**config_wandb)

    # Path to save the figure
    save_path = (
        PATH_POS_LOCAL
        / "experiments"
        / "diagnostics"
        / "visualizations"
        / config["model"]
        / "prior"
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # Acces to ensemble
    fname = PATH_MODEL / config["model"] / "generation" / "prior" / date / "ensemble_prior.pt"
    if not os.path.exists(fname):
        raise FileNotFoundError(f"ERROR - Ensemble not found at {fname}.")

    # Loading data
    x_truth = next(iter(get_dataloaders(batch_size=12)[0]))[0]
    x_ensemble = torch.load(fname, weights_only=True, map_location="cpu")[:, :, 0]

    # Generating mask of the Black Sea
    mask_bs = generate_trajectory_mask(trajectory_size=1)[0]

    # Masking the land
    x_truth[:, mask_bs == 0] = np.nan

    # Extracting variables
    x_truth_oxy, x_truth_chl, x_truth_sal, x_truth_temp, x_truth_ssh = torch.split(
        x_truth[:, :, 0], DATASET_REGION["level"].stop, dim=1
    )

    x_ens_oxy, x_ens_chl, x_ens_sal, x_ens_temp, x_ens_ssh = torch.split(
        x_ensemble, DATASET_REGION["level"].stop, dim=1
    )

    # Extracting depth levels
    levels = xr.open_zarr(PATH_STAT).isel(level=DATASET_REGION["level"]).load().level.values

    for x_gt, x_ens, v in zip(
        [x_truth_oxy, x_truth_chl, x_truth_sal, x_truth_temp],
        [x_ens_oxy, x_ens_chl, x_ens_sal, x_ens_temp],
        DATASET_VARIABLES_OCEAN,
    ):
        for l in range(x_gt.shape[1]):
            #
            # Extracting level data
            x_gt_l, x_ens_l = x_gt[:, l], x_ens[:, l]

            # Creating visualization
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

            # Plotting data
            data = np.concatenate([x_gt_l, x_ens_l], axis=0)
            for i, ax in enumerate(axs.flat):
                q_min, q_max = np.nanquantile(data[i], [0.05, 0.95])
                ax.imshow(np.flipud(data[i]), cmap=CMAPS_SURF[v], vmin=q_min, vmax=q_max)
                ax.axis("off")
                if i == 0:
                    level_str = f" | Level {l} = {levels[l]:.3f} [m]"
                    ax.set_title(f"{TRANSLATION[v]}{level_str}", fontsize=20)

            # Sending to Weights and Biases
            wandb.log({f"PRIOR | {TRANSLATION[v]} / Level {l}": wandb.Image(fig)})

            # Saving locally
            fig.savefig(
                save_path / f"{date}_{TRANSLATION[v]}_l{l}.png",
                bbox_inches="tight",
                dpi=350,
            )

            # Closing the figure
            plt.close(fig)
