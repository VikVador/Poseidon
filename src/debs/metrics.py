# -------------------------------------------------------
#
#        |
#       / \
#      / _ \                  ESA - PROJECT
#     |.o '.|
#     |'._.'|          BLACK SEA DEOXYGENATION EMULATOR
#     |     |
#   ,'|  |  |`.             BY VICTOR MANGELEER
#  /  |  |  |  \
#  |,-'--|--'-.|                2023-2024
#
#
# -------------------------------------------------------
#
# Documentation
# -------------
# A tool to compute a large variety of metrics on the outputs of a model.
#
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter

# Pytorch
import torch
import torch.nn as nn
import torch.optim as optim

# Torch metrics (from Pytorch Lightning)
from torchmetrics.regression     import MeanAbsoluteError, MeanSquaredError, PearsonCorrCoef, R2Score
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall,  BinaryMatthewsCorrCoef


class BlackSea_Metrics():
    r"""A tool to compute a large variety of metrics to assess the quality of a model."""

    def __init__(self, y_true: np.array, y_pred: np.array, treshold_normalized_oxygen: float, treshold_nb_samples: int = 10):

        # Security
        assert y_true.shape == y_pred.shape, f"ERROR (BlackSea_Metrics) - The shapes of the ground truth and predicted values must be the same {y_true.shape} != ({y_pred.shape})."

        # Storing dimensions for ease of comprehension
        self.samples, self.days, self.x, self.y = y_true.shape

        # Storing ground truth and prediction
        self.y_true = y_true
        self.y_pred = y_pred

        # Stores the number of predictions that must be found in a sample to compute metrics, e.g. it can be 0 sometimes !
        self.treshold_nb_samples = treshold_nb_samples

        # Stores the treshold for the normalized oxygen, i.e. the concentration level at which hypoxia is considered
        self.treshold_normalized_oxygen = treshold_normalized_oxygen

        # Metrics used to assess the quality of the model at doing regression (MSE, RMSE, R2, Pearson)
        self.metrics_regression = [MeanSquaredError(),
                                   MeanSquaredError(),
                                   R2Score(),
                                   PearsonCorrCoef()]

        self.metrics_regression_names = ["Mean Squared Error",
                                         "Root Mean Squared Error",
                                         "R2 Score",
                                         "Pearson Correlation Coefficient"]

        # Metrics used to assess the quality of the model at doing classification
        self.metrics_classification = [BinaryAccuracy(),
                                       BinaryPrecision(),
                                       BinaryRecall(),
                                       BinaryMatthewsCorrCoef()]

        self.metrics_classification_names = ["Accuracy",
                                             "Precision",
                                             "Recall",
                                             "Matthews Correlation Coefficient"]

    def get_metrics_names(self):
        r"""Retreives the name of the metrics used for regression and classification"""
        return self.metrics_regression_names + self.metrics_classification_names

    def get_number_of_valid_samples(self):
        r"""Only samples with at least a certain amount of predictions (treshold_nb_samples ) are considered valid."""

        # Create a boolean mask where values are not equal to -1
        mask = self.y_true[:, 0] != -1

        # Count the number of predictions for each sample
        nb_predictions = torch.sum(mask == 1, dim = (1, 2))

        return sum([1 if valid_pred >= self.treshold_nb_samples else 0 for valid_pred in nb_predictions])

    def compute_metrics(self):
        r"""Computes the summed value of a metric using all samples (for each indivudal days), i.e. a tensor of shape (days, metrics value)"""
        return [self.compute_metrics_per_day(self.y_true[:, i], self.y_pred[:, i]) for i in range(self.days)]

    def compute_metrics_average(self, metrics_results: torch.Tensor, number_valid_samples: int):
        r"""Computes the average value of all different metrics for each individual days"""

        # Summing all the results accross batches dimensions, i.e. (batch, days of forecast, metrics) -> (days of forecast, metrics
        metrics_results = torch.sum(metrics_results, dim = 0)

        # Averaging by the number of valid samples
        return metrics_results / number_valid_samples

    def compute_metrics_per_day(self, y_true_per_day: np.array, y_pred_per_day: np.array):
        r"""Computes the summed value of a metric using all samples"""

        # Stores all the results for each metric
        results = []

        # ------------ REGRESSION ------------
        for m_name, m in zip(self.metrics_regression_names, self.metrics_regression):

            # Stores results for the current metric
            results_metric = []

             # Looping over each sample
            for s_true, s_pred in zip(y_true_per_day, y_pred_per_day):

                # Extracting the mask
                mask = s_true != -1

                # Makes sure there is at least two predicted values to compute metric (needed for R2)
                if torch.sum(mask == 1).item() < self.treshold_nb_samples:
                    continue

                # Computing the metric
                if m_name == "Root Mean Squared Error":
                    results_metric.append(torch.sqrt(torch.nan_to_num(m(s_true[mask], s_pred[mask]), nan = 0.0)))
                else:
                    results_metric.append(torch.nan_to_num(m(s_true[mask], s_pred[mask]), nan = 0.0))

            # Storing the sum of all results (it will be normalized by the size of the validation set afterwards)
            results.append(torch.sum(torch.stack(results_metric)))

        # ---------- CLASSIFICATION ----------
        for m_name, m in zip(self.metrics_classification_names, self.metrics_classification):

            # Stores results for the current metric
            results_metric = []

            # Looping over each sample
            for s_true, s_pred in zip(y_true_per_day, y_pred_per_day):

                # Extracting the mask
                mask = s_true != -1

                # Makes sure there is at least two predicted values to compute metric (needed for R2)
                if torch.sum(mask == 1).item() < self.treshold_nb_samples:
                    continue

                # Converting to classification (1 = Hypoxia, 0 = No Hypoxia)
                s_true = (s_true <= self.treshold_normalized_oxygen).type(torch.int)
                s_pred = (s_pred <= self.treshold_normalized_oxygen).type(torch.int)

                # Computing the metric
                results_metric.append(torch.nan_to_num(m(s_true[mask], s_pred[mask]), nan = 0.0))

            # Storing the sum of all results (it will be normalized by the size of the validation set afterwards)
            results.append(torch.sum(torch.stack(results_metric)))

        return results

    def plot_comparison(self, y_true: torch.Tensor, y_pred: torch.Tensor, normalized_deoxygenation_treshold: float, index_sample: int = 0):
        r"Used to visually compare the prediction and the ground truth in wandb"

        # Retrieving the dimensions (ease of comprehension)
        samples, days, x, y = y_pred.shape

        # Creating the figure
        fig, ax = plt.subplots(4, days, figsize = (3 * days, 10))

        # Changing the colormaps
        color_regression     = plt.colormaps["magma"].reversed()
        color_classification = plt.colormaps["viridis"].reversed()

        # Results for classification, i.e. fixed region over time (2)
        for i in range(days):

            # Used to deal with 1D or 2D arrays
            ax_temp = ax[:, i] if days > 1 else ax

            # Retreiving samples and mask (needs to be clone otherwise we have problem with masks)
            y_true_day = torch.clone(y_true[index_sample, i, :, :])
            y_pred_day = torch.clone(y_pred[index_sample, i, :, :])
            mask       = torch.clone(y_true[index_sample, i, :, :] == -1)

            # Hiding the land (for ease of visualization and also, the neural network cannot predict anything on the land using -1 values)
            y_pred_day[mask == True] = 10
            y_true_day[mask == True] = 10

            # Plotting regression problem
            im1 = ax_temp[0].imshow(y_pred_day, cmap = color_regression, vmin = 0, vmax = 1)
            im2 = ax_temp[1].imshow(y_true_day, cmap = color_regression, vmin = 0, vmax = 1)

            # Highligthing the regions where there is hypoxia
            y_pred_class = (y_pred_day < normalized_deoxygenation_treshold).long()
            y_true_class = (y_true_day < normalized_deoxygenation_treshold).long()

            # Hiding the land ()
            y_pred_class[mask == True] = -1
            y_true_class[mask == True] = -1

            # Plotting classification problem
            im3 = ax_temp[2].imshow(y_pred_class, cmap = color_classification, vmin = -1, vmax = 1)
            im4 = ax_temp[3].imshow(y_true_class, cmap = color_classification, vmin = -1, vmax = 1)

            # Adding more information to the plot
            ax_temp[0].set_title(f"Day = {i}", fontsize = 11)

            for j in range(4):
                if i != 0:
                    ax_temp[j].set_yticks([])
                if j != 3:
                    ax_temp[j].set_xticks([])

        def format_classification_ticks(value, _):
            r"Used to format the ticks of the colorbar"
            if value == 1:
                return 'Hypoxia'
            elif value == 0:
                return 'Oxygenated'
            elif value == -1:
                return 'Land'
            else:
                return ''

        # Adding colorbar (regression)
        cbar = fig.colorbar(im1, ax = ax[:2], orientation = 'vertical', fraction = 0.1, pad = 0.05)
        cbar.set_label('Normalized Oxygen Concentration [-]', rotation = 270, labelpad = 30, fontsize = 11)

        # Adding colorbar (classification)
        cbar = fig.colorbar(im3, ax = ax[2:], orientation = 'vertical', fraction = 0.1, pad = 0.05)
        cbar_ticks = MaxNLocator(integer = True)
        cbar.set_ticks(cbar_ticks)
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_classification_ticks))

        # Giving back the fig, i.e. to be send to wandb
        return fig