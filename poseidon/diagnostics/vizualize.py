r"""Visualisation tools."""

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import wandb
import xarray as xr

from einops import rearrange
from poseidon.data.const import DATASET_REGION, DATASET_VARIABLES, DATASET_VARIABLES_OCEAN
from typing import Dict, Optional

# isort: split
from poseidon.config import LOCAL, PATH_DATA, PATH_MODEL, PATH_STAT
from poseidon.data.const import (
    DATASET_VARIABLES_SURFACE,
    TOY_DATASET_REGION,
    TOY_DATASET_VARIABLES,
    TOY_DATASET_VARIABLES_OCEAN,
    TOY_DATASET_VARIABLES_SURFACE,
)
from poseidon.data.mappings import from_tensor_to_xarray
from poseidon.diagnostics.const import CMAPS_LINE, CMAPS_SURF, INTERVALS, TRANSLATION, UNITS


def plot_unconditional(config: Dict, config_setup: Dict) -> None:
    r"""Visualizes unconditional nowcasts.

    Arguments:
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

    # Load ground truth and nowcasts
    x_gt = (
        xr.open_zarr(PATH_DATA)
        .sel(time=slice("2020-01-01", "2020-12-31"))
        .isel(**region, time=np.linspace(0, 365, 12, dtype=int))
        .load()
    )
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
            save_path = f"{LOCAL}/poseidon/metrics/visualizations/{config['model']}/prior_illustrations/{v}"
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


def plot_unconditional_distributions(config: Dict, config_setup: Dict) -> None:
    r"""Visualizes unconditional nowcast and ground truth distributions for all variables.

    Arguments:
        config: Model configuration dictionary.
        config_setup: Configuration setup dictionary.
    """

    # Weights & Biases
    wandb.init(
        project=config_setup["wandb_project"],
        mode=config_setup["wandb_mode"],
        name=config["model"],
        resume="allow",
    )

    # Initializations
    toy_problem = config["toy_problem"]
    region = TOY_DATASET_REGION if toy_problem else DATASET_REGION
    variables = TOY_DATASET_VARIABLES if toy_problem else DATASET_VARIABLES
    variables_surf = TOY_DATASET_VARIABLES_SURFACE if toy_problem else DATASET_VARIABLES_SURFACE
    variables_ocean = TOY_DATASET_VARIABLES_OCEAN if toy_problem else DATASET_VARIABLES_OCEAN
    folder = PATH_MODEL / config["model"] / "nowcasts" / "unconditional"

    # Load ground truth and nowcasts
    x_gt = (
        xr.open_zarr(PATH_DATA)
        .sel(time=slice("1995-01-01", "2017-12-31"))
        .isel(**region, time=np.linspace(0, 365 * 21, 256, dtype=int))
        .load()
    )
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
    x = from_tensor_to_xarray(x=x, variables=variables, region=region)

    # Unscaling the data
    stats = xr.open_zarr(PATH_STAT).isel(level=TOY_DATASET_REGION["level"]).load()
    x = x * stats.sel(statistic="std") + stats.sel(statistic="mean")
    x_gt = x_gt * stats.sel(statistic="std") + stats.sel(statistic="mean")

    def smooth_hist(
        data: np.ndarray,
        bins: np.ndarray,
        sigma: float = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Helper function to smooth histogram data."""
        counts, edges = np.histogram(data, bins=bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        kernel = np.exp(-0.5 * (np.linspace(-3, 3, 7) / sigma) ** 2)
        kernel /= kernel.sum()
        smooth = np.convolve(counts, kernel, mode="same")
        return centers, smooth

    def plot_variable(
        v: str,
        x_gt_values: np.ndarray,
        x_values: np.ndarray,
        bins: np.ndarray,
        levels: np.ndarray,
        is_surface: bool = False,
    ):
        r"""Helper function to plot the distribution of a variable."""
        if is_surface:
            fig, ax = plt.subplots(figsize=(7, 1))
            x1, y1 = smooth_hist(x_gt_values, bins)
            x2, y2 = smooth_hist(x_values, bins)
            ax.plot(x1, y1, color="black", linewidth=1.5, label="$P (X)$")
            ax.fill_between(x2, y2, color=CMAPS_LINE[v], alpha=0.75, label="$P_{\\theta}(X)$")
            ax.set_yticks([])
            for spine in ["top", "right", "left"]:
                ax.spines[spine].set_visible(False)
            ax.spines["bottom"].set_visible(True)
            ax.text(
                bins[0],
                0.2,
                f"{levels[0]:.2f} [m]",
                fontsize=7,
                fontweight="bold",
                va="bottom",
                ha="left",
                transform=ax.get_xaxis_transform(),
            )
            ax.set_xlabel(f"{TRANSLATION[v]} {UNITS[v]}", fontsize=12)
            ax.legend(
                loc="upper left", bbox_to_anchor=(0, 2.25), fontsize=12, frameon=False, ncol=2
            )
            wandb.log({f"PRIOR | Distributions / {TRANSLATION[v]} ": wandb.Image(fig)})
            if config_setup["saving"]:
                save_path = f"{LOCAL}/poseidon/metrics/visualizations/{config['model']}/prior_distributions/"
                os.makedirs(save_path, exist_ok=True)
                fig.savefig(f"{save_path}/{v}.png", bbox_inches="tight", dpi=512)
            plt.close(fig)
        else:
            L = x_gt_values.shape[0]
            fig, axes = plt.subplots(L, 1, figsize=(7, 0.8 * L), sharex=True)
            for i in range(L):
                ax = axes[i]
                x1, y1 = smooth_hist(x_gt_values[i], bins)
                x2, y2 = smooth_hist(x_values[i], bins)
                ax.plot(x1, y1, color="black", linewidth=1.5, label="$P (X)$")
                ax.fill_between(x2, y2, color=CMAPS_LINE[v], alpha=0.75, label="$P_{\\theta}(X)$")
                ax.set_yticks([])
                for spine in ["top", "right", "left"]:
                    ax.spines[spine].set_visible(False)
                ax.spines["bottom"].set_visible(True)
                ax.text(
                    bins[0],
                    0.2,
                    f"{levels[i]:.2f} [m]",
                    fontsize=7,
                    fontweight="bold",
                    va="bottom",
                    ha="left",
                    transform=ax.get_xaxis_transform(),
                )
                if i != L - 1:
                    ax.xaxis.set_visible(False)
                else:
                    ax.set_xlabel(f"{TRANSLATION[v]} {UNITS[v]}", fontsize=12)
                if i == 0:
                    ax.legend(
                        loc="upper left",
                        bbox_to_anchor=(0, 2.25),
                        fontsize=12,
                        frameon=False,
                        ncol=2,
                    )
            plt.tight_layout()
            wandb.log({f"PRIOR | Distributions / {TRANSLATION[v]} ": wandb.Image(fig)})
            if config_setup["saving"]:
                save_path = f"{LOCAL}/poseidon/metrics/visualizations/{config['model']}/prior_distributions/"
                os.makedirs(save_path, exist_ok=True)
                fig.savefig(f"{save_path}/{v}.png", bbox_inches="tight", dpi=512)
            plt.close(fig)

    # Ocean
    for v in variables_ocean:
        x_gt_values = rearrange(x_gt[v].values, "B Z X Y -> Z (B X Y)")
        x_values = rearrange(x[v].values, "B Z K X Y -> Z (B K X Y)")
        levels = stats.level.values
        bins = np.linspace(INTERVALS[v][0], INTERVALS[v][1], 128)
        plot_variable(v, x_gt_values, x_values, bins, levels, is_surface=False)

    # Surface variables (single line)
    for v in variables_surf:
        x_gt_values = x_gt[v].values[:, :, :, 0].flatten()
        x_values = x[v].values[:, :, :, :, 0].flatten()
        levels = stats.level.values
        bins = np.linspace(INTERVALS[v][0], INTERVALS[v][1], 128)
        plot_variable(v, x_gt_values, x_values, bins, levels, is_surface=True)
