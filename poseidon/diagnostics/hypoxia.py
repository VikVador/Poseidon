r"""Tools to determine correction hypoxia treshold."""

import numpy as np
import torch
import xarray as xr

from pathlib import Path

# isort: split
from poseidon.config import PATH_DATA, PATH_MODEL, PATH_STAT
from poseidon.data.const import TOY_DATASET_REGION, TOY_DATASET_VARIABLES
from poseidon.data.datasets import PoseidonDataset
from poseidon.data.mappings import from_tensor_to_indices
from poseidon.data.mask import generate_trajectory_mask
from poseidon.diagnostics.const import DATES_POSTERIOR
from poseidon.diagnostics.metrics import next_day
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score
from torch import Tensor
from typing import (
    Dict,
    Sequence,
)


def from_tensor_to_xarray_modified(
    x: Tensor,
    variables: Sequence[str],
    region: Dict[str, slice],
    path: Path = PATH_DATA,
) -> xr.Dataset:
    r"""Transform a (batch of) stacked tensor into an :class:`Xarray dataset`.

    Arguments:
        x: Input tensor (C, T, X, Y).
        variables: Variable present in the stacked tensor.
        region: Region used to extract the data from original dataset.
        path: Path to the original dataset.
    """
    assert 4 <= x.ndim < 7, "ERROR - Input tensor must have shape (C, T, X, Y)"
    while x.ndim < 6:
        x = x.unsqueeze(dim=0)

    # Extracting data associated to each variable
    data_slices = {
        v: x[:, :, idx_start:idx_end]
        for v, (idx_start, idx_end) in from_tensor_to_indices(
            path=path,
            variables=variables,
            region=region,
        ).items()
    }

    # Creating Xarray dataset
    data_arrays = []
    for v, data in data_slices.items():
        data_array = xr.DataArray(
            data=data,
            dims=("samples", "batch", "level", "trajectory", "latitude", "longitude"),
            name=v,
        )

        if data_array.shape[2] == 1:
            data_array = data_array.squeeze(dim="level")
        data_arrays.append(data_array)

    return xr.merge(data_arrays)


def compute_classification_metrics(truth: torch.Tensor, members: torch.Tensor):
    r"""Computing classification metrics"""

    # Stores average results
    acc, pre, rec = [], [], []

    # Number of levels
    samples, ensembles, levels = members.shape[0], members.shape[1], members.shape[2]

    # Looping over each sample
    for s in range(samples):
        print(f"Sample {s}/{samples}")

        # Stores results per sample
        s_acc, s_pre, s_rec = [], [], []

        for e in range(ensembles):
            print(f"Ensemble {e}/{ensembles}")

            # Stores results per ensemble
            e_acc, e_pre, e_rec = [], [], []

            # Computing for each level
            for l in range(levels):
                # Extracting data
                y_true = truth[s, :, l].flatten()
                y_pred = members[s, :, l].flatten()

                # Creating joint mask
                mask_joint = ~np.isnan(y_true) & ~np.isnan(y_pred)

                # Extracting values
                y_true = y_true[mask_joint]
                y_pred = y_pred[mask_joint]

                # Computing metrics
                if len(np.unique(y_true)) == 1:
                    e_acc.append(1.0 if np.array_equal(y_true, y_pred) else 0.0)
                else:
                    e_acc.append(balanced_accuracy_score(y_true, y_pred))
                e_pre.append(precision_score(y_true, y_pred, zero_division=np.nan, labels=[0, 1]))
                e_rec.append(recall_score(y_true, y_pred, zero_division=np.nan, labels=[0, 1]))

            # Creating tensors
            e_acc, e_pre, e_rec = np.stack(e_acc), np.stack(e_pre), np.stack(e_rec)

            # Adding results
            s_acc.append(e_acc)
            s_pre.append(e_pre)
            s_rec.append(e_rec)

        # Creating tensors
        s_acc, s_pre, s_rec = np.stack(s_acc), np.stack(s_pre), np.stack(s_rec)

        # Adding results
        acc.append(s_acc)
        pre.append(s_pre)
        rec.append(s_rec)

    # Creating tensors
    acc = np.stack(acc)
    pre = np.stack(pre)
    rec = np.stack(rec)

    return torch.tensor(acc), torch.tensor(pre), torch.tensor(rec)


def determine_threshold():
    # ==================
    #    Loading Data
    # ==================
    #
    # Configuration
    config = {"model": "laced-puddle-18"}

    # Hypoxia threshold [mmol/m³]
    hypoxia_threshold = 63

    # Stores multiples results
    x_posterior_ground_truth, x_posterior_d_theta = list(), list()

    # Looping over dates
    for i in range(24):
        # Extracting the date
        date = DATES_POSTERIOR[i]

        # Displaying information over terminal
        print(f"Processing date: {date}")

        # P(x_d)
        x_pgt, _ = next(
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

        # P(X|d, y)_theta
        x_pdt = torch.load(
            PATH_MODEL
            / config["model"]
            / "nowcasts"
            / "conditional"
            / date
            / "nowcast_conditional.pt",
            weights_only=False,
            map_location=torch.device("cpu"),
        )

        # Storing results
        x_posterior_ground_truth.append(x_pgt)
        x_posterior_d_theta.append(x_pdt)

    # Stacking results
    x_posterior_ground_truth = torch.stack(x_posterior_ground_truth, dim=0)
    x_posterior_d_theta = torch.stack(x_posterior_d_theta, dim=0)

    # Mask of the Black Sea
    mask = generate_trajectory_mask(
        variables=TOY_DATASET_VARIABLES,
        region=TOY_DATASET_REGION,
        trajectory_size=1,
    )

    # Masking Data
    x_posterior_ground_truth[:, mask[0] == 0] = np.nan
    x_posterior_d_theta[:, :, mask[0] == 0] = np.nan

    # Converting to xarray dataset for final touch of processing
    ds_posterior_ground_truth = from_tensor_to_xarray_modified(
        x_posterior_ground_truth, variables=TOY_DATASET_VARIABLES, region=TOY_DATASET_REGION
    )
    ds_posterior_with_obs_NN = from_tensor_to_xarray_modified(
        x_posterior_d_theta, variables=TOY_DATASET_VARIABLES, region=TOY_DATASET_REGION
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

    # Extracting oxygen nowcast
    posterior_oxygen_ground_truth = ds_posterior_ground_truth["DOX"].values[:, :, :, 0]
    posterior_oxygen_with_obs_NN = ds_posterior_with_obs_NN["DOX"].values[:, :, :, 0]

    print("GT:", posterior_oxygen_ground_truth.shape)
    print("NN:", posterior_oxygen_with_obs_NN.shape)

    # Stores the results
    res_acc, res_pre, res_rec, tresh = [], [], [], []

    # Evaluating different threshold correction
    for bias in np.linspace(0, 100, 16):
        # Corrected threshold
        hypoxia_threshold_corrected = hypoxia_threshold + bias

        # Treshold used
        tresh.append(hypoxia_threshold_corrected)

        # Detecting hypoxia
        x_gt = (posterior_oxygen_ground_truth < hypoxia_threshold) * 1
        x_nn = (posterior_oxygen_with_obs_NN < hypoxia_threshold) * 1

        # Broadcasting for ease
        x_gt = np.swapaxes(x_gt, 0, 1)
        x_gt = np.repeat(x_gt, 64, axis=1)

        # Computing metrics
        acc, pre, rec = compute_classification_metrics(x_gt, x_nn)

        # Appending results
        res_acc.append(acc)
        res_pre.append(pre)
        res_rec.append(rec)

    # Stacking the results
    res_acc, res_pre, res_rec = torch.stack(res_acc), torch.stack(res_pre), torch.stack(res_rec)

    # Computing average accross members
    acc, pre, rec = (
        torch.nanmean(res_acc, dim=2),
        torch.nanmean(res_pre, dim=2),
        torch.nanmean(res_rec, dim=2),
    )

    # Computing average accross samples
    acc, pre, rec = torch.nanmean(acc, dim=1), torch.nanmean(pre, dim=1), torch.nanmean(rec, dim=1)

    # Final shapes
    print("Final Accuracy shape:", acc.shape)
    print("Final Precision shape:", pre.shape)
    print("Final Recall shape:", rec.shape)
    print("Tresholds:", tresh)

    # Saving results
    torch.save(
        acc, "/gpfs/home/acad/ulg-mast/vmangele/poseidon/metrics/data/classification/acc.pt"
    )
    torch.save(
        pre, "/gpfs/home/acad/ulg-mast/vmangele/poseidon/metrics/data/classification/pre.pt"
    )
    torch.save(
        rec, "/gpfs/home/acad/ulg-mast/vmangele/poseidon/metrics/data/classification/rec.pt"
    )
    torch.save(
        torch.from_numpy(np.stack(tresh)),
        "/gpfs/home/acad/ulg-mast/vmangele/poseidon/metrics/data/classification/tresh.pt",
    )
