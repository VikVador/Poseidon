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


def visualize_distance(dates: list, config: Dict, config_wandb: Dict) -> None:
    r"""Visualizes distances metrics between ensembles.

    Arguments:
        dates: List of ensemble dates (YYYY-MM-DD).
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
        / "distance"
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # Access to distance
    paths, path_distance = [], PATH_MODEL / config["model"] / "diagnostics" / "distance"
    for d in dates:
        if not os.path.exists(path_distance / d / "wasserstein.pt"):
            continue
        else:
            paths.append(path_distance / d / "wasserstein.pt")
    distances = torch.stack(
        [torch.load(p, weights_only=True, map_location="cpu") for p in paths], dim=0
    )

    # Extracting variables
    dis_oxy, dis_chl, dis_sal, dis_temp, dis_ssh = torch.split(
        distances, DATASET_REGION["level"].stop, dim=1
    )

    # Extracting depth levels
    levels = xr.open_zarr(PATH_STAT).isel(level=DATASET_REGION["level"]).load().level.values

    # Creating visualization
    fig, axs = plt.subplots(1, 4, figsize=(12, 6), sharey=True)
    colors, labels = (
        [
            "#5e4c5f",
            "#a00000",
            "#ffbb6f",
        ],
        ["$P(X|d)$", "$P_{\\theta}(X|d, y)$", "$P(X)$"],
    )

    for i, (dis, v) in enumerate(
        zip([dis_oxy, dis_chl, dis_sal, dis_temp], DATASET_VARIABLES_OCEAN)
    ):
        # Extracting distances
        dis_prior_x_d, dis_prior_x_d_theta, dis_prior_x = torch.split(dis, 1, dim=2)

        for arr, color, label in zip(
            [dis_prior_x_d, dis_prior_x_d_theta, dis_prior_x], colors, labels
        ):
            arr = arr.squeeze()
            quantiles = torch.tensor([0.25, 0.5, 0.75], dtype=arr.dtype, device=arr.device)
            Q1, Q2, Q3 = torch.quantile(arr, quantiles, dim=0).cpu().numpy()
            axs[i].plot(Q2, levels, color=color, lw=2, label=f"{label}")
            axs[i].fill_betweenx(levels, Q1, Q3, color=color, alpha=0.2)

        axs[i].set_xlim(0)
        axs[i].set_yscale("log")
        axs[i].set_title(TRANSLATION[v])
        axs[i].grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.6)
        if i > 0:
            axs[i].set_ylabel("")
        axs[i].set_xlabel("")

    # Common ylabel
    axs[0].set_ylabel("Depth [m]")

    # Common xlabel centered between 2nd and 3rd subplot
    fig.text(0.5, -0.03, "Wasserstein Distance", fontsize=18, ha="center", va="center")

    handles, legend_labels = axs[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper left",
        bbox_to_anchor=(0.95, 0.93),
        borderaxespad=0.0,
        fontsize=18,
        frameon=True,
    )

    # Depth decreasing from top to bottom
    plt.gca().invert_yaxis()
    plt.tight_layout()

    # Sending to Weights and Biases & Saving locally
    wandb.log({"PRIOR | Distances / Wasserstein": wandb.Image(fig)})

    # Saving locally
    fig.savefig(
        save_path / "distance.png",
        bbox_inches="tight",
        dpi=350,
    )

    # Closing the figure
    plt.close(fig)
