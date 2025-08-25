r"""Tools to compute metrics for the paper."""

import numpy as np
import os
import torch
import xarray as xr

from datetime import datetime, timedelta
from scipy.stats import wasserstein_distance

# isort: split
from poseidon.config import PATH_DATA, PATH_MODEL, PATH_STAT
from poseidon.data.const import (
    TOY_DATASET_REGION,
    TOY_DATASET_VARIABLES,
    TOY_DATASET_VARIABLES_OCEAN,
    TOY_DATASET_VARIABLES_SURFACE,
)
from poseidon.data.dataloaders import get_toy_dataloaders
from poseidon.data.datasets import PoseidonDataset
from poseidon.data.mappings import from_tensor_to_xarray
from poseidon.diagnostics.const import TRANSLATION


def next_day(date: str):
    r"""Helper tool to determine date of following day."""
    date_obj = datetime.strptime(date, "%Y-%m-%d")
    next_date_obj = date_obj + timedelta(days=1)
    return next_date_obj.strftime("%Y-%m-%d")


def computing_metrics_prior(date: str, config: dict):
    r"""Computes prior metrics for used in paper."""

    # ==================
    #   Loading Data
    # ==================
    #
    # P(X)
    dl_train, _, _ = get_toy_dataloaders(batch_size=config["p(x)_samples"])
    x_prior, _ = next(iter(dl_train))

    # P(X|d)
    dates_start, dates_end = (
        [f"{year}-{date[5:]}" for year in range(1995, 1998)],
        [next_day(f"{year}-{date[5:]}") for year in range(1995, 1998)],
    )

    x_prior_d = []
    for ds, de in zip(dates_start, dates_end):
        dataset = PoseidonDataset(
            path=PATH_DATA,
            date_start=ds,
            date_end=de,
            variables=TOY_DATASET_VARIABLES,
            region=TOY_DATASET_REGION,
        )

        sample, _ = next(iter(dataset))
        x_prior_d.append(sample)

    x_prior_d = torch.stack(x_prior_d)

    # P(X|d)_1 & P(X|d)_2
    idx_even, idx_odd = (
        [i for i in range(x_prior_d.shape[0]) if i % 2 == 0],
        [i for i in range(x_prior_d.shape[0]) if i % 2 == 1],
    )

    x_prior_d_even, x_prior_d_odd = (x_prior_d[idx_even].clone(), x_prior_d[idx_odd].clone())

    # P(X|d)_theta
    x_prior_d_theta = torch.load(
        PATH_MODEL
        / config["model"]
        / "nowcasts"
        / "unconditional"
        / date
        / "nowcast_unconditional.pt",
        weights_only=False,
    )

    # ==================
    #    Wasserstein
    # ==================
    #
    # Transforming the data to xarray
    ds_x_prior = from_tensor_to_xarray(
        x_prior,
        variables=TOY_DATASET_VARIABLES,
        region=TOY_DATASET_REGION,
    )
    ds_x_prior_d = from_tensor_to_xarray(
        x_prior_d,
        variables=TOY_DATASET_VARIABLES,
        region=TOY_DATASET_REGION,
    )
    ds_x_prior_d_even = from_tensor_to_xarray(
        x_prior_d_even,
        variables=TOY_DATASET_VARIABLES,
        region=TOY_DATASET_REGION,
    )
    ds_x_prior_d_odd = from_tensor_to_xarray(
        x_prior_d_odd,
        variables=TOY_DATASET_VARIABLES,
        region=TOY_DATASET_REGION,
    )
    ds_x_prior_d_theta = from_tensor_to_xarray(
        x_prior_d_theta,
        variables=TOY_DATASET_VARIABLES,
        region=TOY_DATASET_REGION,
    )

    for v in TOY_DATASET_VARIABLES_OCEAN:
        # Displaying information over terminal
        print(f"Processing variable: {v}")

        # Extracting the associated data (removing time axis)
        v_x_prior = ds_x_prior[v].values[:, :, 0]
        v_x_prior_d = ds_x_prior_d[v].values[:, :, 0]
        v_x_prior_d_even = ds_x_prior_d_even[v].values[:, :, 0]
        v_x_prior_d_odd = ds_x_prior_d_odd[v].values[:, :, 0]
        v_x_prior_d_theta = ds_x_prior_d_theta[v].values[:, :, 0]

        # Displaying information over terminal
        print(f"Analyzing Variable: {TRANSLATION[v]}")
        print(f"  prior_v shape: {v_x_prior.shape}")
        print(f"  prior_daily_v shape: {v_x_prior_d.shape}")
        print(f"  prior_daily_v_even shape: {v_x_prior_d_even.shape}")
        print(f"  prior_daily_v_odd shape: {v_x_prior_d_odd.shape}")
        print(f"  prior_daily_NN_v shape: {v_x_prior_d_theta.shape}")

        # Stores Wasserstein distances
        v_z_wd = []

        for z in range(2):
            # Displaying information over terminal
            print(f" |- Analyzing level {z}...")

            # Extracting the associated data
            v_z_x_prior = v_x_prior[:, z].flatten()
            v_z_x_prior_d = v_x_prior_d[:, z].flatten()
            v_z_x_prior_d_even = v_x_prior_d_even[:, z].flatten()
            v_z_x_prior_d_odd = v_x_prior_d_odd[:, z].flatten()
            v_z_x_prior_d_theta = v_x_prior_d_theta[:, z].flatten()

            # Removing NaNs from the data
            v_z_x_prior = v_z_x_prior[~np.isnan(v_z_x_prior)]
            v_z_x_prior_d = v_z_x_prior_d[~np.isnan(v_z_x_prior_d)]
            v_z_x_prior_d_odd = v_z_x_prior_d_odd[~np.isnan(v_z_x_prior_d_odd)]
            v_z_x_prior_d_even = v_z_x_prior_d_even[~np.isnan(v_z_x_prior_d_even)]
            v_z_x_prior_d_theta = v_z_x_prior_d_theta[~np.isnan(v_z_x_prior_d_theta)]

            v_z_wd.append(
                torch.tensor([
                    wasserstein_distance(
                        v_z_x_prior_d_even, v_z_x_prior_d_odd
                    ),  # D( P(X|d) & P(X|d) )
                    wasserstein_distance(
                        v_z_x_prior_d, v_z_x_prior_d_theta
                    ),  # D( P(X|d) & P(X|d)_theta )
                    wasserstein_distance(v_z_x_prior_d, v_z_x_prior),  # D( P(X|d) & P(X) )
                ]).unsqueeze(0)
            )

        # Path to folder in which save the data
        f_save = f"/gpfs/home/acad/ulg-mast/vmangele/poseidon/metrics/data/unconditional/{date}/"
        if not os.path.exists(f_save):
            os.makedirs(f_save)

        # Saving the data
        torch.save(torch.cat(v_z_wd, dim=0), f_save + f"{v}.pt")

    for v in TOY_DATASET_VARIABLES_SURFACE:
        # Displaying information over terminal
        print(f"Processing variable: {v}")

        # Extracting the associated data (removing time axis)
        v_x_prior = ds_x_prior[v].values[:, 0]
        v_x_prior_d = ds_x_prior_d[v].values[:, 0]
        v_x_prior_d_even = ds_x_prior_d_even[v].values[:, 0]
        v_x_prior_d_odd = ds_x_prior_d_odd[v].values[:, 0]
        v_x_prior_d_theta = ds_x_prior_d_theta[v].values[:, 0]

        # Displaying information over terminal
        print(f"Analyzing Variable: {TRANSLATION[v]}")
        print(f"  prior_v shape: {v_x_prior.shape}")
        print(f"  prior_daily_v shape: {v_x_prior_d.shape}")
        print(f"  prior_daily_v_even shape: {v_x_prior_d_even.shape}")
        print(f"  prior_daily_v_odd shape: {v_x_prior_d_odd.shape}")
        print(f"  prior_daily_NN_v shape: {v_x_prior_d_theta.shape}")

        # Stores Wasserstein distances
        v_z_wd = []

        # Extracting the associated data
        v_z_x_prior = v_x_prior.flatten()
        v_z_x_prior_d = v_x_prior_d.flatten()
        v_z_x_prior_d_even = v_x_prior_d_even.flatten()
        v_z_x_prior_d_odd = v_x_prior_d_odd.flatten()
        v_z_x_prior_d_theta = v_x_prior_d_theta.flatten()

        # Removing NaNs from the data
        v_z_x_prior = v_z_x_prior[~np.isnan(v_z_x_prior)]
        v_z_x_prior_d = v_z_x_prior_d[~np.isnan(v_z_x_prior_d)]
        v_z_x_prior_d_odd = v_z_x_prior_d_odd[~np.isnan(v_z_x_prior_d_odd)]
        v_z_x_prior_d_even = v_z_x_prior_d_even[~np.isnan(v_z_x_prior_d_even)]
        v_z_x_prior_d_theta = v_z_x_prior_d_theta[~np.isnan(v_z_x_prior_d_theta)]

        v_z_wd.append(
            torch.tensor([
                wasserstein_distance(
                    v_z_x_prior_d_even, v_z_x_prior_d_odd
                ),  # D( P(X|d) & P(X|d) )
                wasserstein_distance(
                    v_z_x_prior_d, v_z_x_prior_d_theta
                ),  # D( P(X|d) & P(X|d)_theta )
                wasserstein_distance(v_z_x_prior_d, v_z_x_prior),  # D( P(X|d) & P(X) )
            ]).unsqueeze(0)
        )

        # Path to folder in which save the data
        f_save = f"/gpfs/home/acad/ulg-mast/vmangele/poseidon/metrics/data/unconditional/{date}/"
        if not os.path.exists(f_save):
            os.makedirs(f_save)

        # Saving the data
        torch.save(torch.cat(v_z_wd, dim=0), f_save + f"{v}.pt")


def computing_metrics_posterior(date: str, config: dict):
    r"""Computes prior metrics for used in paper."""

    # ==================
    #   Loading Data
    # ==================
    #
    # P(x_d)
    x_posterior_ground_truth, _ = next(
        iter(
            PoseidonDataset(
                path=PATH_DATA,
                date_start=date,
                date_end=next_day(date),
                variables=TOY_DATASET_VARIABLES,
                region=TOY_DATASET_REGION,
            )
        )
    )

    # P(X|d)_theta
    x_prior_d_theta = torch.load(
        PATH_MODEL
        / config["model"]
        / "nowcasts"
        / "unconditional"
        / f"2017-{date[5:]}"
        / "nowcast_unconditional.pt",
        weights_only=False,
        map_location=torch.device("cpu"),
    )

    # P(X|d, y)_theta
    x_posterior_d_theta = torch.load(
        PATH_MODEL
        / config["model"]
        / "nowcasts"
        / "conditional"
        / date
        / "nowcast_conditional.pt",
        weights_only=False,
        map_location=torch.device("cpu"),
    )

    # =======================================================
    # COMPUTING SPREAD/SKILL RATIO FOR INDIVIDUAL VARIABLES
    # =======================================================
    # Transforming the data to xarray
    ds_posterior_ground_truth = from_tensor_to_xarray(
        x_posterior_ground_truth, variables=TOY_DATASET_VARIABLES, region=TOY_DATASET_REGION
    )
    ds_posterior_with_obs_NN = from_tensor_to_xarray(
        x_posterior_d_theta, variables=TOY_DATASET_VARIABLES, region=TOY_DATASET_REGION
    )
    ds_posterior_without_obs_NN = from_tensor_to_xarray(
        x_prior_d_theta, variables=TOY_DATASET_VARIABLES, region=TOY_DATASET_REGION
    )

    # Loading statistics
    stats = xr.open_zarr(PATH_STAT).isel(level=TOY_DATASET_REGION["level"]).load()

    # Unscaling the data to physical units
    ds_posterior_ground_truth = ds_posterior_ground_truth * stats.sel(statistic="std") + stats.sel(
        statistic="mean"
    )
    ds_posterior_with_obs_NN = ds_posterior_with_obs_NN * stats.sel(statistic="std") + stats.sel(
        statistic="mean"
    )
    ds_posterior_without_obs_NN = ds_posterior_without_obs_NN * stats.sel(
        statistic="std"
    ) + stats.sel(statistic="mean")

    for v in TOY_DATASET_VARIABLES_OCEAN:
        # Extracting the associated data (removing time axis)
        posterior_ground_truth_v = ds_posterior_ground_truth[v].values[:, :, 0]
        posterior_with_obs_NN_v = ds_posterior_with_obs_NN[v].values[:, :, 0]
        posterior_without_obs_NN_v = ds_posterior_without_obs_NN[v].values[:, :, 0]

        # Displaying information over terminal
        print(f"Analyzing Variable: {TRANSLATION[v]}")
        print(f"  posterior_ground_truth_v shape: {posterior_ground_truth_v.shape}")
        print(f"  posterior_with_obs_NN_v shape: {posterior_with_obs_NN_v.shape}")
        print(f"  posterior_without_obs_NN_v shape: {posterior_without_obs_NN_v.shape}")

        metrics_SPREAD = np.concatenate(
            [
                np.nanmean(
                    np.nanstd(posterior_with_obs_NN_v, axis=(0), keepdims=True, ddof=1),
                    axis=(2, 3),
                ).swapaxes(0, 1),
                np.nanmean(
                    np.nanstd(posterior_without_obs_NN_v, axis=(0), keepdims=True, ddof=1),
                    axis=(2, 3),
                ).swapaxes(0, 1),
            ],
            axis=1,
        )

        # Computing the mean nowcast
        posterior_with_obs_NN_v = np.nanmean(posterior_with_obs_NN_v, axis=0, keepdims=True)
        posterior_without_obs_NN_v = np.nanmean(posterior_without_obs_NN_v, axis=0, keepdims=True)

        metrics_SKILL = np.sqrt(
            np.nanmean(
                np.concatenate(
                    [
                        (posterior_ground_truth_v - posterior_with_obs_NN_v) ** 2,
                        (posterior_ground_truth_v - posterior_without_obs_NN_v) ** 2,
                    ],
                    axis=0,
                ),
                axis=(2, 3),
            )
        ).swapaxes(0, 1)

        # Path to folder in which save the data
        f_save = f"/gpfs/home/acad/ulg-mast/vmangele/poseidon/metrics/data/conditional/{date}/{v}/"
        if not os.path.exists(f_save):
            os.makedirs(f_save)

        # Saving the data
        torch.save(metrics_SKILL, f_save + "skill.pt")
        torch.save(metrics_SPREAD, f_save + "spread.pt")

    for v in TOY_DATASET_VARIABLES_SURFACE:
        # Extracting the associated data (removing time axis)
        posterior_ground_truth_v = ds_posterior_ground_truth[v].values[:, 0, :, :, 0]
        posterior_with_obs_NN_v = ds_posterior_with_obs_NN[v].values[:, 0, :, :, 0]
        posterior_without_obs_NN_v = ds_posterior_without_obs_NN[v].values[:, 0, :, :, 0]

        # Displaying information over terminal
        print(f"Analyzing Variable: {TRANSLATION[v]}")
        print(f"  posterior_ground_truth_v shape: {posterior_ground_truth_v.shape}")
        print(f"  posterior_with_obs_NN_v shape: {posterior_with_obs_NN_v.shape}")
        print(f"  posterior_without_obs_NN_v shape: {posterior_without_obs_NN_v.shape}")

        metrics_SPREAD = np.concatenate(
            [
                np.nanmean(
                    np.nanstd(posterior_with_obs_NN_v, axis=(0), keepdims=True, ddof=1),
                    axis=(1, 2),
                    keepdims=True,
                )[0],
                np.nanmean(
                    np.nanstd(posterior_without_obs_NN_v, axis=(0), keepdims=True, ddof=1),
                    axis=(1, 2),
                    keepdims=True,
                )[0],
            ],
            axis=0,
        ).swapaxes(0, 1)

        # Computing the mean nowcast
        posterior_with_obs_NN_v = np.nanmean(posterior_with_obs_NN_v, axis=0, keepdims=True)
        posterior_without_obs_NN_v = np.nanmean(posterior_without_obs_NN_v, axis=0, keepdims=True)

        metrics_SKILL = np.sqrt(
            np.nanmean(
                np.concatenate(
                    [
                        (posterior_ground_truth_v - posterior_with_obs_NN_v) ** 2,
                        (posterior_ground_truth_v - posterior_without_obs_NN_v) ** 2,
                    ],
                    axis=0,
                ),
                axis=(1, 2),
            )
        )[:, None].swapaxes(0, 1)

        # Path to folder in which save the data
        f_save = f"/gpfs/home/acad/ulg-mast/vmangele/poseidon/metrics/data/conditional/{date}/{v}/"
        if not os.path.exists(f_save):
            os.makedirs(f_save)

        # Saving the data
        torch.save(metrics_SKILL, f_save + "skill.pt")
        torch.save(metrics_SPREAD, f_save + "spread.pt")
