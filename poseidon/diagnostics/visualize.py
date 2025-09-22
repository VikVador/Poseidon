r"""Tools to generate ensembles."""

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import wandb
import xarray as xr

from typing import Dict
from zipfile import Path

# isort: split
from poseidon.config import PATH_MODEL, PATH_POS_LOCAL, PATH_STAT
from poseidon.data.const import DATASET_REGION, DATASET_VARIABLES_OCEAN
from poseidon.data.dataloaders import get_dataloaders
from poseidon.data.mask import generate_trajectory_mask
from poseidon.diagnostics import HYPOXIA_THRESHOLDS
from poseidon.diagnostics.const import CMAPS_LINE, CMAPS_SURF, TRANSLATION, UNITS


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

            # Sending to Weights and Biases & Saving locally
            wandb.log({f"PRIOR | {TRANSLATION[v]} / Level {l}": wandb.Image(fig)})
            fig.savefig(
                save_path / f"{date}_{TRANSLATION[v]}_l{l}.png",
                bbox_inches="tight",
                dpi=350,
            )

            # Closing
            plt.close(fig)
            wandb.finish()


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
    fig.savefig(
        save_path / "distance.png",
        bbox_inches="tight",
        dpi=350,
    )

    # Closing
    plt.close(fig)
    wandb.finish()


def visualize_denoiser(config: Dict, config_wandb: Dict) -> None:
    r"""Visualizes samples from E[x|xt] for different noise levels.

    Arguments:
        dates: List of ensemble dates (MM-DD).
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
        / "denoiser"
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    save_path_training = save_path / "training"
    if not os.path.exists(save_path_training):
        os.makedirs(save_path_training, exist_ok=True)

    save_path_validation = save_path / "validation"
    if not os.path.exists(save_path_validation):
        os.makedirs(save_path_validation, exist_ok=True)

    # Access to main folder
    path_folder = PATH_MODEL / config["model"] / "denoising"

    # Loading data
    x_train_truth = torch.load(
        path_folder / "training" / "sample_truth.pt",
        map_location="cpu",
        weights_only=True,
    )
    x_train_noisy = torch.load(
        path_folder / "training" / "sample_noisy.pt",
        map_location="cpu",
        weights_only=True,
    )
    x_train_recon = torch.load(
        path_folder / "training" / "sample_reconstruction.pt",
        map_location="cpu",
        weights_only=True,
    )
    x_valid_truth = torch.load(
        path_folder / "validation" / "sample_truth.pt",
        map_location="cpu",
        weights_only=True,
    )
    x_valid_noisy = torch.load(
        path_folder / "validation" / "sample_noisy.pt",
        map_location="cpu",
        weights_only=True,
    )
    x_valid_recon = torch.load(
        path_folder / "validation" / "sample_reconstruction.pt",
        map_location="cpu",
        weights_only=True,
    )
    noise_levels = torch.load(
        path_folder / "training" / "noise_levels.pt",
        map_location="cpu",
        weights_only=True,
    )

    # Generating mask of the Black Sea
    mask_bs = generate_trajectory_mask(trajectory_size=1)[0]

    # Masking the data
    x_train_truth[mask_bs == 0] = np.nan
    x_train_noisy[:, mask_bs == 0] = np.nan
    x_valid_truth[mask_bs == 0] = np.nan
    x_valid_noisy[:, mask_bs == 0] = np.nan

    # Visualizing
    for n, noise in enumerate(noise_levels):
        train_qmin, train_qmax = (
            torch.nanquantile(x_train_truth[0, 0], 0.02),
            torch.nanquantile(x_train_truth[0, 0], 0.98),
        )
        valid_qmin, valid_qmax = (
            torch.nanquantile(x_valid_truth[0, 0], 0.02),
            torch.nanquantile(x_valid_truth[0, 0], 0.98),
        )

        fig, axes = plt.subplots(1, 3, figsize=(20, 20))
        axes[0].imshow(
            np.flipud(x_train_truth[0, 0, :, :]), cmap="inferno", vmin=train_qmin, vmax=train_qmax
        )
        axes[1].imshow(
            np.flipud(x_train_noisy[n, 0, 0, :, :]),
            cmap="inferno",
            vmin=train_qmin,
            vmax=train_qmax,
        )
        axes[2].imshow(
            np.flipud(x_train_recon[n, 0, 0, :, :]),
            cmap="inferno",
            vmin=train_qmin,
            vmax=train_qmax,
        )

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)

        labels = ["TRUTH", rf"NOISY | $\sigma$ = {noise:.6f}", "RECONSTRUCTION"]
        paddings = [-11, -12, -11]
        for ax, label, padding in zip(axes, labels, paddings):
            ax.text(
                3.1,
                padding,
                label,
                color="white",
                fontsize=10,
                fontweight="bold",
                verticalalignment="top",
                horizontalalignment="left",
                bbox=dict(facecolor="black", edgecolor="none", pad=5),
            )

        plt.tight_layout()

        # Sending to Weights and Biases & Saving locally
        wandb.log({"DENOISER | Reconstructions / Training": wandb.Image(fig)})
        fig.savefig(
            save_path / "training" / f"reconstruction_{noise:.6f}.png",
            bbox_inches="tight",
            dpi=350,
        )

        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(20, 20))
        axes[0].imshow(
            np.flipud(x_valid_truth[0, 0, :, :]), cmap="inferno", vmin=valid_qmin, vmax=valid_qmax
        )
        axes[1].imshow(
            np.flipud(x_valid_noisy[n, 0, 0, :, :]),
            cmap="inferno",
            vmin=valid_qmin,
            vmax=valid_qmax,
        )
        axes[2].imshow(
            np.flipud(x_valid_recon[n, 0, 0, :, :]),
            cmap="inferno",
            vmin=valid_qmin,
            vmax=valid_qmax,
        )

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)

        labels = ["TRUTH", rf"NOISY | $\sigma$ = {noise:.6f}", "RECONSTRUCTION"]
        paddings = [-11, -12, -11]
        for ax, label, padding in zip(axes, labels, paddings):
            ax.text(
                3.1,
                padding,
                label,
                color="white",
                fontsize=10,
                fontweight="bold",
                verticalalignment="top",
                horizontalalignment="left",
                bbox=dict(facecolor="black", edgecolor="none", pad=5),
            )

        plt.tight_layout()

        # Sending to Weights and Biases & Saving locally
        wandb.log({"DENOISER | Reconstructions / Validation": wandb.Image(fig)})

        # Saving locally
        fig.savefig(
            save_path / "validation" / f"reconstruction_{noise:.6f}.png",
            bbox_inches="tight",
            dpi=350,
        )

        plt.close(fig)

    wandb.finish()


def visualize_spread_skill_ratio(dates: list, config: Dict, config_wandb: Dict) -> None:
    r"""Visualizes evolution of the spread skill ratio with respect to depth.

    Arguments:
        dates: List of ensemble dates (YYYY-MM-DD).
        config: Configuration for generation.
        config_wandb: Configuration setup dictionary.
    """

    def _load_files_for_dates(path_folder: Path, dates: list, filename: str) -> torch.Tensor:
        r"""Helper tool to load files for multiple dates."""
        loaded_tensors = []
        for date in dates:
            file_path = path_folder / date / filename
            if file_path.exists():
                loaded_tensors.append(torch.load(file_path, weights_only=True, map_location="cpu"))
        return torch.stack(loaded_tensors, dim=0)[:, :128]

    # fmt: off
    # Initialization of Weights and Biases
    wandb.init(**config_wandb)

    # Path to save the figure
    save_path = PATH_POS_LOCAL / "experiments" / "diagnostics" / "visualizations" / config["model"] / "ssr"
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # Access to main folder
    path_folder = PATH_MODEL / config["model"] / "diagnostics" / "spread_skill"

    # Loading data
    skill_prior      = _load_files_for_dates(path_folder, dates, "skill_prior.pt")
    skill_posterior  = _load_files_for_dates(path_folder, dates, "skill_posterior.pt")
    spread_prior     = _load_files_for_dates(path_folder, dates, "spread_prior.pt")
    spread_posterior = _load_files_for_dates(path_folder, dates, "spread_posterior.pt")
    ssr_prior        = _load_files_for_dates(path_folder, dates, "ssr_prior.pt")
    ssr_posterior    = _load_files_for_dates(path_folder, dates, "ssr_posterior.pt")

    # Extracting depth levels
    levels = xr.open_zarr(PATH_STAT).isel(level=DATASET_REGION["level"]).load().level.values

    # Visualization
    fig, axes = plt.subplots(3, 4, figsize=(14, 14), sharey=True)

    for i, var in enumerate(DATASET_VARIABLES_OCEAN):

        # Extracting variables
        skill_prior_v      = skill_prior[:, i * 32 : (i + 1) * 32]
        skill_posterior_v  = skill_posterior[:, i * 32 : (i + 1) * 32]
        spread_prior_v     = spread_prior[:, i * 32 : (i + 1) * 32]
        spread_posterior_v = spread_posterior[:, i * 32 : (i + 1) * 32]
        ssr_prior_v        = ssr_prior[:, i * 32 : (i + 1) * 32]
        ssr_posterior_v    = ssr_posterior[:, i * 32 : (i + 1) * 32]

        # Computing statistics
        quantiles = torch.tensor([0.25, 0.5, 0.75], dtype=skill_prior.dtype)

        skill_prior_q1, skill_prior_q2, skill_prior_q3                = torch.nanquantile(skill_prior_v, quantiles, dim=0)
        skill_posterior_q1, skill_posterior_q2, skill_posterior_q3    = torch.nanquantile(skill_posterior_v, quantiles, dim=0)
        spread_prior_q1, spread_prior_q2, spread_prior_q3             = torch.nanquantile(spread_prior_v, quantiles, dim=0)
        spread_posterior_q1, spread_posterior_q2, spread_posterior_q3 = torch.nanquantile(spread_posterior_v, quantiles, dim=0)
        ssr_prior_q1, ssr_prior_q2, ssr_prior_q3                      = torch.nanquantile(ssr_prior_v, quantiles, dim=0)
        ssr_posterior_q1, ssr_posterior_q2, ssr_posterior_q3          = torch.nanquantile(ssr_posterior_v, quantiles, dim=0)

        # Plotting Skill
        axes[0, i].plot(skill_prior_q2,     levels, label="Prior",     color="grey",          linestyle="--")
        axes[0, i].plot(skill_posterior_q2, levels, label="Posterior", color=CMAPS_LINE[var], linestyle="-")

        axes[0, i].fill_betweenx(levels, skill_prior_q1,     skill_prior_q3,     color="grey",          alpha=0.3)
        axes[0, i].fill_betweenx(levels, skill_posterior_q1, skill_posterior_q3, color=CMAPS_LINE[var], alpha=0.3)

        axes[0, i].set_title(f"{TRANSLATION[var]} ${UNITS[var]}$", fontsize=12)
        axes[0, i].set_yscale("log")
        axes[0, i].set_ylim(levels.min(), levels.max())
        axes[0, i].invert_yaxis()
        axes[0, i].grid(True, which="both", linestyle=":")
        axes[0, i].set_xlim(left=0, right=axes[0, i].get_xlim()[1])
        if i == 0:
            axes[0, i].set_ylabel("Depth [m]", fontsize=12)
        if i == 3:
            axes[0, i].set_ylabel("Skill", rotation=270, labelpad=20, verticalalignment='bottom', horizontalalignment='center', fontweight='semibold')
            axes[0, i].yaxis.set_label_position('right')

        # Plotting Spread
        axes[1, i].plot(spread_prior_q2,     levels, label="Prior",     color="grey",          linestyle="--")
        axes[1, i].plot(spread_posterior_q2, levels, label="Posterior", color=CMAPS_LINE[var], linestyle="-")

        axes[1, i].fill_betweenx(levels, spread_prior_q1,     spread_prior_q3,     color="grey",          alpha=0.3)
        axes[1, i].fill_betweenx(levels, spread_posterior_q1, spread_posterior_q3, color=CMAPS_LINE[var], alpha=0.3)

        axes[1, i].set_yscale("log")
        axes[1, i].set_ylim(levels.min(), levels.max())
        axes[1, i].invert_yaxis()
        axes[1, i].grid(True, which="both", linestyle=":")
        axes[1, i].set_xlim(left=0, right=axes[1, i].get_xlim()[1])
        if i == 0:
            axes[1, i].set_ylabel("Depth [m]")
        if i == 3:
            axes[1, i].set_ylabel("Spread", rotation=270, labelpad=20, verticalalignment='bottom', horizontalalignment='center', fontweight='semibold')
            axes[1, i].yaxis.set_label_position('right')

        # Plotting Spread Skill Ratio
        axes[2, i].plot(ssr_prior_q2,     levels, label="Prior",     color="grey",          linestyle="--")
        axes[2, i].plot(ssr_posterior_q2, levels, label="Posterior", color=CMAPS_LINE[var], linestyle="-")

        axes[2, i].fill_betweenx(levels, ssr_prior_q1,     ssr_prior_q3,     color="grey",          alpha=0.3)
        axes[2, i].fill_betweenx(levels, ssr_posterior_q1, ssr_posterior_q3, color=CMAPS_LINE[var], alpha=0.3)

        axes[2, i].set_yscale("log")
        axes[2, i].set_ylim(levels.min(), levels.max())
        axes[2, i].grid(True, which="both", linestyle=":")
        axes[2, i].invert_yaxis()
        axes[2, i].axvline(x=1.0, linestyle=":")
        axes[2, i].set_xlim([0, 2])
        if i == 0:
            axes[2, i].set_ylabel("Depth [m]")
        if i == 3:
            axes[2, i].set_ylabel("SSR Ratio [-]", rotation=270, labelpad=20, verticalalignment='bottom', horizontalalignment='center', fontweight='semibold')
            axes[2, i].yaxis.set_label_position('right')


    # Sending to Weights and Biases & Saving locally
    plt.tight_layout()
    wandb.log({"POSTERIOR | Spread-Skill-Ratio / Results": wandb.Image(fig)})
    fig.savefig(
        save_path / "spread_skill_ratio.png",
        bbox_inches="tight",
        dpi=350,
    )

    plt.close(fig)
    wandb.finish()


def visualize_hypoxia(dates: list, config: Dict, config_wandb: Dict) -> None:
    r"""Visualizes results of classification metrics for hypoxia detection.

    Arguments:
        dates: List of ensemble dates (YYYY-MM-DD).
        config: Configuration for generation.
        config_wandb: Configuration setup dictionary.
    """

    # fmt: off
    # Initialization of Weights and Biases
    wandb.init(**config_wandb)

    # Path to save the figure
    save_path = PATH_POS_LOCAL / "experiments" / "diagnostics" / "visualizations" / config["model"] / "hypoxia"
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # Access to main folder
    path_folder = PATH_MODEL / config["model"] / "diagnostics" / "classification"

    # Loading data
    accuracy  = torch.stack([torch.stack([torch.load(path_folder / date / f"accuracy_{int(tresh)}.pt",  weights_only=True, map_location="cpu") for tresh in HYPOXIA_THRESHOLDS], dim = 0) for date in dates], dim = 0)
    precision = torch.stack([torch.stack([torch.load(path_folder / date / f"precision_{int(tresh)}.pt", weights_only=True, map_location="cpu") for tresh in HYPOXIA_THRESHOLDS], dim = 0) for date in dates], dim = 0)
    recall    = torch.stack([torch.stack([torch.load(path_folder / date / f"recall_{int(tresh)}.pt",    weights_only=True, map_location="cpu") for tresh in HYPOXIA_THRESHOLDS], dim = 0) for date in dates], dim = 0)
    f1        = torch.stack([torch.stack([torch.load(path_folder / date / f"f1_{int(tresh)}.pt",        weights_only=True, map_location="cpu") for tresh in HYPOXIA_THRESHOLDS], dim = 0) for date in dates], dim = 0)
    roc_auc   = torch.stack([torch.stack([torch.load(path_folder / date / f"roc_auc_{int(tresh)}.pt",   weights_only=True, map_location="cpu") for tresh in HYPOXIA_THRESHOLDS], dim = 0) for date in dates], dim = 0)

    # Extracting depth levels
    levels = xr.open_zarr(PATH_STAT).isel(level=DATASET_REGION["level"]).load().level.values

    # Computing statistics
    variables = {
        "Accuracy [%]": accuracy.nanmean(dim=0),
        "Precision [%]": precision.nanmean(dim=0),
        "Recall [%]": recall.nanmean(dim=0),
        "F1 Score [%]": f1.nanmean(dim=0),
        "ROC-AUC Score [%]": roc_auc.nanmean(dim=0)
    }

    # Visualization
    fig, axes = plt.subplots(1, 5, figsize=(20, 8), sharey=True)
    for ax, (var_name, var_mean) in zip(axes, variables.items()):
        for thresh_idx in range(len(HYPOXIA_THRESHOLDS)):
            ax.plot(var_mean[thresh_idx].numpy() * 100, levels, label=f"{HYPOXIA_THRESHOLDS[thresh_idx]:.0f}")
        ax.set_title(var_name, fontweight="semibold", pad=8)
        ax.set_xlim(0, 100)
        ax.grid(True)
        ax.invert_yaxis()
        ax.set_yscale("log")

    axes[0].set_ylabel("Depth [m]")

    # Common legend
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc='center left', bbox_to_anchor=(0.98, 0.88), fontsize='small', title=f"Thresholds ${UNITS['DOX']}$", title_fontsize='medium')

    # fmt: on
    # Sending to Weights and Biases & Saving locally
    wandb.log({"POSTERIOR | Hypoxia Detection / Results": wandb.Image(fig)})
    plt.tight_layout()
    fig.savefig(
        save_path / "results.png",
        bbox_inches="tight",
        dpi=350,
    )

    plt.close(fig)
    wandb.finish()
