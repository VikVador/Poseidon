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

        self.metrics_regression_names = ["Mean Squared Error (Average Per Sample)",
                                         "Root Mean Squared Error (Average Per Sample)",
                                         "R2 Score (Average Per Sample)",
                                         "Pearson Correlation Coefficient (Average Per Sample)"]

        # Metrics used to assess the quality of the model at doing classification
        self.metrics_classification = [BinaryAccuracy(),
                                       BinaryPrecision(),
                                       BinaryRecall(),
                                       BinaryMatthewsCorrCoef()]

        self.metrics_classification_names = ["Accuracy (Average Per Sample)",
                                             "Precision (Average Per Sample)",
                                             "Recall (Average Per Sample)",
                                             "Matthews Correlation Coefficient (Average Per Sample)"]

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
                if m_name == "Root Mean Squared Error (Average Per Sample)":
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