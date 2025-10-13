r"""Tools to compute diagnostics."""

import numpy as np
import os
import torch
import xarray as xr

from einops import rearrange
from scipy.stats import wasserstein_distance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from typing import Dict

# fmt: off
#
# isort: split
from poseidon.config import PATH_DATA, PATH_MODEL, PATH_STAT
from poseidon.data.const import (
    DATASET_DATES_TRAINING,
    DATASET_REGION,
    DATASET_VARIABLES,
    DATASET_VARIABLES_OCEAN,
)
from poseidon.data.dataloaders import get_dataloaders
from poseidon.data.datasets import PoseidonDataset
from poseidon.data.mask import generate_trajectory_mask


def compute_spread_skill(date: str, config: Dict) -> None:
    r"""Computes the spread-skill ratio between an ensemble and the truth.

    Arguments:
        date: Ensemble date (YYYY-MM-DD).
        config: Configuration for computing metric.
    """

    # Access path to save results
    path_spread_skill = PATH_MODEL / config["model"] / "diagnostics" / "spread_skill" / date
    if not os.path.exists(path_spread_skill):
            os.makedirs(path_spread_skill)

    # Access path to ensembles
    path_ensemble_prior     = PATH_MODEL / config["model"] / "generation" / "prior"     / date[5:] / "ensemble_prior.pt"
    path_ensemble_posterior = PATH_MODEL / config["model"] / "generation" / "posterior" / date     / "ensemble_posterior.pt"

    # Generates a mask of the Black Sea
    mask_bs = generate_trajectory_mask(trajectory_size=1)[0]

    # Loading the data
    x_ens_prior     = torch.load(path_ensemble_prior,     weights_only=True, map_location="cpu")
    x_ens_posterior = torch.load(path_ensemble_posterior, weights_only=True, map_location="cpu")
    x_truth, _      = next(iter(PoseidonDataset(
                            path=PATH_DATA,
                            date_start=date,
                            date_end=date)))

    # Extracting statistics for unscaling data
    ds_mean = xr.open_zarr(PATH_STAT).isel(level=DATASET_REGION["level"]).sel(statistic="mean")[DATASET_VARIABLES].load()
    ds_std  = xr.open_zarr(PATH_STAT).isel(level=DATASET_REGION["level"]).sel(statistic="std" )[DATASET_VARIABLES].load()
    mean    = torch.concat([torch.from_numpy(ds_mean[var].values) for var in DATASET_VARIABLES_OCEAN] + [torch.tensor([ds_mean["ssh"].values[0]])], dim=0)[None, :, None, None, None]
    std     = torch.concat([torch.from_numpy(ds_std[var].values)  for var in DATASET_VARIABLES_OCEAN] + [torch.tensor([ds_std["ssh"].values[0]])],  dim=0)[None, :, None, None, None]

    # Unscaling the data
    x_truth         = x_truth         * std[0] + mean[0]
    x_ens_prior     = x_ens_prior     * std    + mean
    x_ens_posterior = x_ens_posterior * std    + mean

    # Masking the data
    x_truth[mask_bs == 0]            = np.nan
    x_ens_prior[:, mask_bs == 0]     = np.nan
    x_ens_posterior[:, mask_bs == 0] = np.nan

    # Computing spread
    spread_prior     = torch.sqrt(torch.nanmean(torch.var(x_ens_prior,     dim=0, correction=1), dim=(1, 2, 3)))
    spread_posterior = torch.sqrt(torch.nanmean(torch.var(x_ens_posterior, dim=0, correction=1), dim=(1, 2, 3)))

    # Computing ensemble mean
    x_ens_prior_mean = torch.nanmean(x_ens_prior, dim=0)
    x_ens_post_mean  = torch.nanmean(x_ens_posterior, dim=0)

    # Computing skill
    skill_prior     = torch.sqrt(torch.nanmean((x_truth - x_ens_prior_mean).pow(2), dim=(1, 2, 3)))
    skill_posterior = torch.sqrt(torch.nanmean((x_truth - x_ens_post_mean).pow(2),  dim=(1, 2, 3)))

    # Number of ensemble members
    M_prior, M_posterior = x_ens_prior.shape[0], x_ens_posterior.shape[0]

    # Computing spread-skill ratio
    ssr_prior     = torch.sqrt(torch.tensor(1 + 1 / M_prior))     * (spread_prior     / skill_prior)
    ssr_posterior = torch.sqrt(torch.tensor(1 + 1 / M_posterior)) * (spread_posterior / skill_posterior)

    # Saving the results
    torch.save(ssr_prior,        path_spread_skill / "ssr_prior.pt")
    torch.save(skill_prior,      path_spread_skill / "skill_prior.pt")
    torch.save(spread_prior,     path_spread_skill / "spread_prior.pt")
    torch.save(ssr_posterior,    path_spread_skill / "ssr_posterior.pt")
    torch.save(skill_posterior,  path_spread_skill / "skill_posterior.pt")
    torch.save(spread_posterior, path_spread_skill / "spread_posterior.pt")

def compute_distance(date: str, config: Dict) -> None:
    r"""Computes a distance metric between multiple distributions.

    Distributions:
        P(X|d), P_θ(X|d) and P(X).

    Arguments:
        date: Prior distribution date (MM-DD).
        config: Configuration for computing distance metric.
    """

    # Access path to save results
    path_distance = PATH_MODEL / config["model"] / "diagnostics" / "distance" / date
    if not os.path.exists(path_distance):
            os.makedirs(path_distance)

    # Access path to ensembles
    path_ensemble_prior = PATH_MODEL / config["model"] / "generation" / "prior" / date / "ensemble_prior.pt"

    # Mask of the Black Sea
    mask_bs = generate_trajectory_mask(trajectory_size=1)[0]

    # Extracting training years bounds
    year_start, year_end = int(DATASET_DATES_TRAINING[0][:4]), int(DATASET_DATES_TRAINING[1][:4])

    # Stores P(X|d)
    x_prior_d = []

    for y in range(year_start, year_end):

        # Current date of sample
        current_date = f"{y}-{date}"

        # Adding sample
        x_prior_d.append(next(iter(PoseidonDataset(path=PATH_DATA, date_start=current_date, date_end=current_date)))[0])

    # Loading samples
    x_prior_d       = torch.stack(x_prior_d, dim = 0)
    x_prior         = next(iter(get_dataloaders(batch_size = 1024)[0]))[0]
    x_prior_d_theta = torch.load(path_ensemble_prior, weights_only=True, map_location="cpu")

    # Sub-sampling distributions
    x_prior_d_even, x_prior_d_odd = x_prior_d[::2], x_prior_d[1::2]

    # Masking the land
    x_prior[:, mask_bs == 0]         = np.nan
    x_prior_d_even[:, mask_bs == 0]  = np.nan
    x_prior_d_odd[:, mask_bs == 0]   = np.nan
    x_prior_d_theta[:, mask_bs == 0] = np.nan

    # Stores distance metric
    distances = []

    for l in range(x_prior.shape[1]):

        # Extracting level features
        x_prior_l, x_prior_d_l, x_prior_d_even_l, x_prior_d_odd_l, x_prior_d_theta_l = (
            x_prior[:, l].flatten(),
            x_prior_d[:, l].flatten(),
            x_prior_d_even[:, l].flatten(),
            x_prior_d_odd[:, l].flatten(),
            x_prior_d_theta[:, l].flatten(),
        )

        # Removing NaNs
        x_prior_l, x_prior_d_l, x_prior_d_even_l, x_prior_d_odd_l, x_prior_d_theta_l = (
            x_prior_l[~torch.isnan(x_prior_l)],
            x_prior_d_l[~torch.isnan(x_prior_d_l)],
            x_prior_d_even_l[~torch.isnan(x_prior_d_even_l)],
            x_prior_d_odd_l[~torch.isnan(x_prior_d_odd_l)],
            x_prior_d_theta_l[~torch.isnan(x_prior_d_theta_l)],
        )

        distances.append(torch.tensor([
            wasserstein_distance(x_prior_d_even_l, x_prior_d_odd_l),   # D(P(X|d), P(X|d))
            wasserstein_distance(x_prior_d_l,      x_prior_d_theta_l), # D(P(X|d), P_θ(X|d))
            wasserstein_distance(x_prior_d_l,      x_prior_l),         # D(P(X|d), P(X))
        ]))

    # Saving distances
    torch.save(torch.stack(distances, dim=0), path_distance / "wasserstein.pt")

def compute_hypoxia_classification(date: str, threshold: float, config: Dict) -> None:
    r"""Computes classification metrics on ensemble for hypoxia detection problem.

    Arguments:
        date: Ensemble date (YYYY-MM-DD).
        threshold: Hypoxia detection threshold [mmol/m^3].
        config: Configuration for computing metric.
    """

    # Access path to save results
    path_classification = PATH_MODEL / config["model"] / "diagnostics" / "classification" / date
    if not os.path.exists(path_classification):
            os.makedirs(path_classification)

    # Access path to ensembles
    path_ensemble_posterior = PATH_MODEL / config["model"] / "generation" / "posterior" / date / "ensemble_posterior.pt"

    # Loading statistics
    stats = xr.open_zarr(PATH_STAT).isel(level=DATASET_REGION["level"]).load()

    # Extracting DOX statistics
    dox_mean, dox_std = (
        stats["DOX"].sel(statistic = "mean").values,
        stats["DOX"].sel(statistic = "std").values,
    )

    # Unscaling thresholds (GT from Grégoire, M., V. Garçon et al. (2021). )
    threshold_unscaled = torch.from_numpy((threshold - dox_mean) / dox_std)
    threshold_truth    = torch.from_numpy((63.0      - dox_mean) / dox_std)

    # Generates a mask of the Black Sea
    mask_bs = generate_trajectory_mask(trajectory_size=1)[0]

    # Loading the data
    x_ens_posterior = torch.load(path_ensemble_posterior, weights_only=True, map_location="cpu")
    x_truth, _      = next(iter(PoseidonDataset(
                            path=PATH_DATA,
                            date_start=date,
                            date_end=date)))

    # Masking the data
    x_truth[mask_bs == 0]            = np.nan
    x_ens_posterior[:, mask_bs == 0] = np.nan

    # Extracting oxygen feature
    x_truth = x_truth[:32]
    x_ens_posterior = x_ens_posterior[:, :32]

    # Scaling truth to ensemble size
    x_truth = torch.stack([x_truth for _ in range(x_ens_posterior.shape[0])], dim=0)

    # Flattening the data
    x_truth         = rearrange(x_truth,         "E Z ... -> E Z (...)")
    x_ens_posterior = rearrange(x_ens_posterior, "E Z ... -> E Z (...)")

    # Stores global metrics
    accuracy, accuracy_blc, precision, recall, f1, tpr, fpr = [], [], [], [], [], [], []

    for l in range(x_truth.shape[1]):

        # Extracting level features
        x_truth_l, x_ens_posterior_l = x_truth[:, l], x_ens_posterior[:, l]

        # Stores metrics
        accuracy_l, accuracy_blc_l, precision_l, recall_l, f1_l, tpr_l, fpr_l = [], [], [], [], [], [], []

        for e in range(x_truth.shape[0]):

            # Extracting ensemble member
            x_truth_le, x_ens_posterior_le = x_truth_l[e], x_ens_posterior_l[e]

            # Removing NaNs using common mask
            common_mask = ~torch.isnan(x_truth_le) & ~torch.isnan(x_ens_posterior_le)
            x_truth_le         = x_truth_le[common_mask]
            x_ens_posterior_le = x_ens_posterior_le[common_mask]

            # Applying threshold
            x_truth_le         = (x_truth_le         < threshold_truth[l]).float().cpu()
            x_ens_posterior_le = (x_ens_posterior_le < threshold_unscaled[l]).float().cpu()

            # True Positive, False Positive, True Negative, False Negative counts
            TP = ((x_ens_posterior_le == 1) & (x_truth_le == 1)).sum().item()
            FP = ((x_ens_posterior_le == 1) & (x_truth_le == 0)).sum().item()
            TN = ((x_ens_posterior_le == 0) & (x_truth_le == 0)).sum().item()
            FN = ((x_ens_posterior_le == 0) & (x_truth_le == 1)).sum().item()

            # Compute rates, avoiding division by zero
            TPR = TP / (TP + FN) if (TP + FN) > 0 else np.nan
            FPR = FP / (FP + TN) if (FP + TN) > 0 else np.nan

            # Computing metrics
            precision_l.append(precision_score(x_truth_le, x_ens_posterior_le, zero_division=np.nan, labels=[0, 1]))
            recall_l.append(recall_score(      x_truth_le, x_ens_posterior_le, zero_division=np.nan, labels=[0, 1]))
            f1_l.append(f1_score(              x_truth_le, x_ens_posterior_le, zero_division=np.nan, labels=[0, 1]))
            tpr_l.append(TPR)
            fpr_l.append(FPR)

            # Security for accuracy
            if len(torch.unique(x_truth_le)) == 1:
                accuracy_l.append(1.0 if torch.equal(x_truth_le, x_ens_posterior_le) else 0.0)
                accuracy_blc_l.append(1.0 if torch.equal(x_truth_le, x_ens_posterior_le) else 0.0)
            else:
                accuracy_l.append(accuracy_score(x_truth_le, x_ens_posterior_le))
                accuracy_blc_l.append(balanced_accuracy_score(x_truth_le, x_ens_posterior_le, adjusted=False))

        # Computing mean metrics
        accuracy.append(    torch.nanmean(torch.tensor(accuracy_l)))
        accuracy_blc.append(torch.nanmean(torch.tensor(accuracy_blc_l)))
        precision.append(   torch.nanmean(torch.tensor(precision_l)))
        recall.append(      torch.nanmean(torch.tensor(recall_l)))
        f1.append(          torch.nanmean(torch.tensor(f1_l)))
        tpr.append(         torch.nanmean(torch.tensor(tpr_l)))
        fpr.append(         torch.nanmean(torch.tensor(fpr_l)))

    # Saving results
    torch.save(torch.tensor(accuracy),     path_classification / f"accuracy_{int(threshold)}.pt")
    torch.save(torch.tensor(accuracy_blc), path_classification / f"accuracy_balanced_{int(threshold)}.pt")
    torch.save(torch.tensor(precision),    path_classification / f"precision_{int(threshold)}.pt")
    torch.save(torch.tensor(recall),       path_classification / f"recall_{int(threshold)}.pt")
    torch.save(torch.tensor(f1),           path_classification / f"f1_{int(threshold)}.pt")
    torch.save(torch.tensor(tpr),          path_classification / f"tpr_{int(threshold)}.pt")
    torch.save(torch.tensor(fpr),          path_classification / f"fpr_{int(threshold)}.pt")
