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
import matplotlib.gridspec as gridspec

# Pytorch
import torch
import torch.nn as nn
import torch.optim as optim

# Torch metrics (from Pytorch Lightning)
from torchmetrics.regression     import MeanAbsoluteError, MeanSquaredError, PearsonCorrCoef, R2Score
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryMatthewsCorrCoef, BinaryROC, BinaryAUROC

# Definition of new metrics
def PercentageOfBias(y_pred : np.array = None, y_true : np.array = None):
    r"""Used to compute the percentage of bias"""
    return lambda y_true, y_pred  : np.nanmean((y_true - y_pred) / np.abs(y_true))

def PercentageOfBiasPerPixel(y_pred : np.array = None, y_true : np.array = None):
    r"""Used to compute the percentage of bias"""
    return lambda y_true, y_pred  : np.nanmean((y_true - y_pred) / np.abs(y_true), axis = 0)

def RootMeanSquaredError(y_pred : np.array = None, y_true : np.array = None):
    r"""Used to compute the root mean squared error"""
    return lambda y_true, y_pred  : np.sqrt(np.nanmean((y_true - y_pred) ** 2))

def RootMeanSquaredErrorPerPixel(y_pred : np.array = None, y_true : np.array = None):
    r"""Used to compute the root mean squared error"""
    return lambda y_true, y_pred  : np.sqrt(np.nanmean((y_true - y_pred) ** 2, axis = 0))
class BlackSea_Metrics():
    r"""A tool to compute a large variety of metrics to assess the quality of a model."""

    def __init__(self, mode: str, mask : np.array, treshold : float, number_of_batches : int):
        r"""Initialization of the tool"""

        # Storing information
        self.mask = mask
        self.mode = mode
        self.number_of_batches = number_of_batches
        self.treshold_normalized_oxygen = treshold

        # Used to store results and plots
        self.scores, self.scores_names, self.plots = None, None, list()

        # Definition of the metrics
        self.metrics_regression = [MeanSquaredError(),
                                   RootMeanSquaredError(),
                                   R2Score(),
                                   PearsonCorrCoef(),
                                   PercentageOfBias()]

        self.metrics_classification = [BinaryAccuracy(),
                                       BinaryPrecision(),
                                       BinaryRecall(),
                                       BinaryMatthewsCorrCoef()]

        # Definition of the nams
        self.metrics_regression_names = ["Mean Squared Error",
                                         "Root Mean Squared Error",
                                         "R2 Score",
                                         "Pearson Correlation Coefficient",
                                         "Percentage of Bias"]

        self.metrics_classification_names = ["Accuracy",
                                             "Precision",
                                             "Recall",
                                             "Matthews Correlation Coefficient"]

    def get_names_metrics(self):
        r"""Retreives the name of all the metrics"""
        return      ["Area Under The Curve"] + self.metrics_classification_names if self.mode == "classification" else \
               self.metrics_regression_names + self.metrics_classification_names

    def get_results(self):
        r"""Retreives the results of all the metrics"""
        return self.scores

    def get_plots(self):
        r"""Retreives the plots of all the metrics"""
        return self.plots

    def compute_metrics(self, y_pred: np.array, y_true: np.array):
        r"""Computes each metric for each individual days, i.e. returns a tensor of shape[days, metrics]"""

        # Retrieving dimensions for ease of comprehension
        batch_size, days, = y_true.shape[0], y_true.shape[1]

        # Stores results for each days (temporarily)
        scores_temporary = [self.compute_metrics_(y_true[:, i], y_pred[:, i], i) for i in range(days)]

        # Converting to numpy matrix and divinding by number of batches, i.e. since we sum all the results, we'll obtain the average over batch
        scores_temporary = np.array(scores_temporary) / self.number_of_batches

        # Adding the results
        self.scores = np.sum([self.scores, scores_temporary], axis = 0) if isinstance(self.scores, (np.ndarray, np.generic)) else \
                      np.array(scores_temporary)

    def compute_metrics_(self, y_true_per_day: np.array, y_pred_per_day: np.array, index_day : np.array):
        r"""Computes all the metrics for a given day"""

        # Stores all the results from the different metrics as well as the name of the metrics used
        results, results_name = list(), list()

        # In classification problem, we have access to probabilities and we can compute the AUC
        if self.mode == "classification":

            # Retreiving values in the sea, swapping axis (t, c, x, y) to (c, t, x, y) and flattening (c, t * x * y)
            y_true_per_day = np.swapaxes(y_true_per_day[:, :, self.mask[:-2, :-2] == 1], 0, 1).reshape(2, -1)
            y_pred_per_day = np.swapaxes(y_pred_per_day[:, :, self.mask[:-2, :-2] == 1], 0, 1).reshape(2, -1)

            # Used th compute the area under the curve
            toolAUC = BinaryAUROC()

            # Computations
            results      += [toolAUC(y_pred_per_day, y_true_per_day).item()] # Tool returns a tensor of shape (1, 1) -> (1)
            results_name += [f"Area Under Curve (D" + str(index_day) + ")"]

            # Reshaping the data, i.e. (classes, t * x * y) to (t * x * y) where C = 0 no hypoxia and C = 1 is hypoxia
            y_true_per_day = np.argmax(y_true_per_day, axis = 0)
            y_pred_per_day = np.argmax(y_pred_per_day, axis = 0)

        # In regression problem, we have access to concentrations
        if self.mode == "regression":

            # Retreiving values in the sea, swapping axis (t, c, x, y) to (c, t, x, y) and flattening (c, t * x * y)
            y_true_per_day = y_true_per_day[:, self.mask[:-2, :-2] == 1].reshape(-1)
            y_pred_per_day = y_pred_per_day[:, self.mask[:-2, :-2] == 1].reshape(-1)

            # Computations
            results      += [m(y_pred_per_day, y_true_per_day).item() for m in self.metrics_regression]
            results_name += [n + " (D" + str(index_day) + ")" for n in self.metrics_regression_names]

            # Reshaping the data, i.e. (t, days, x, y) to (t, days, x, y) in binary values using treshold
            y_true_per_day = (y_true_per_day < self.treshold_normalized_oxygen) * 1
            y_pred_per_day = (y_pred_per_day < self.treshold_normalized_oxygen) * 1

        # Classification metrics are computed whatever the type of the problem
        results      += [m(y_pred_per_day, y_true_per_day).item() for m in self.metrics_classification]
        results_name += [n + " (D" + str(index_day) + ")" for n in self.metrics_classification_names]

        # Returning the results and their corresponding names
        return results

    def compute_plots(self, y_pred: np.array, y_true: np.array):
        r"""Creates each plots for each individual days, i.e. a tensor of shape[days, metrics]"""

        # Looping over each day
        for i in range(y_true.shape[1]):

            # Retrieving corresponding day
            y_true_per_day = y_true[:, i]
            y_pred_per_day = y_pred[:, i]

            # ROCAUC
            if self.mode == "classification":

                # Computing metrics
                self.plots += self.compute_plots_classification_ROCAUC(y_pred_per_day[:, :, self.mask[:-2, :-2] == 1],
                                                                         y_true_per_day[:, :, self.mask[:-2, :-2] == 1],
                                                                         i)
                # Changing problem to non-probabilistic
                y_true_per_day = np.argmax(y_true_per_day, axis = 1)
                y_pred_per_day = np.argmax(y_pred_per_day, axis = 1)

            # Regression
            if self.mode == "regression":

                # Computing metrics
                self.plots += self.compute_plots_regression(y_pred_per_day, y_true_per_day, i)

                # Chaning problem to classification
                y_pred_per_day = (y_pred_per_day < self.treshold_normalized_oxygen) * 1
                y_true_per_day = (y_true_per_day < self.treshold_normalized_oxygen) * 1

            # Classification
            self.plots += self.compute_plots_classification(y_pred_per_day, y_true_per_day, i)

    def make_plots(self, score : np.array, index_day : int, label : str, cmap : str, vminmax : tuple):
        r"""Creates the plots based on the results"""

        # Flipping vertically to show correctly Black Sea (for you my loving oceanographer <3)
        score = np.flipud(score)

        # Removing first lines (its just empty sea)
        score = score[20:, :]

        # Creating the figure
        fig = plt.figure(figsize = (15, 10))

        # Adding a grid
        gs = fig.add_gridspec(3, 4, width_ratios = [1, 1, 0.005, 0.05], height_ratios = [1, 0.5, 0.5])

        # Plotting (1) - Whole map
        ax_top_plot = fig.add_subplot(gs[0, :2])
        im1 = ax_top_plot.imshow(score,
                                 cmap   = cmap,
                                 vmin   = vminmax[0],
                                 vmax   = vminmax[1],
                                 aspect = '0.83')

        # Add min and max values to the top right corner
        min_val = np.nanmin(score)
        max_val = np.nanmax(score)
        ax_top_plot.text(0.97, 0.95, f'Min: {min_val:.4f}\nMax: {max_val:.4f}', transform = ax_top_plot.transAxes,
                        verticalalignment = 'top',
                        horizontalalignment = 'right',
                        color = 'white',
                        fontsize = 8,
                        bbox = dict(boxstyle = 'round', facecolor = 'grey', alpha = 0.4))

        # Plotting (2) - Focusing on top left region
        ax_bottom_plot_1 = fig.add_subplot(gs[1, 0])
        im2 = ax_bottom_plot_1.imshow(score[20:110, 175:275],
                                      cmap   = cmap,
                                      vmin   = vminmax[0],
                                      vmax   = vminmax[1],
                                      aspect = '0.43')

        # Plotting (3) - Focusing on top bottom region
        ax_bottom_plot_2 = fig.add_subplot(gs[1, 1])
        im3 = ax_bottom_plot_2.imshow(score[62:100, 250:470],
                                      cmap   = cmap,
                                      vmin   = vminmax[0],
                                      vmax   = vminmax[1],
                                      aspect = '2.2')

        # Plotting (4) - Focusing on bottom region
        ax_bottom_plot_3 = fig.add_subplot(gs[2, :-1])
        im4 = ax_bottom_plot_3.imshow(score[200:, :500],
                                      cmap   = cmap,
                                      vmin   = vminmax[0],
                                      vmax   = vminmax[1],
                                      aspect = '2.35')

        # Plotting (5) - Colorbar
        ax_colorbar = fig.add_subplot(gs[:, -1])
        cbar = plt.colorbar(im1, cax = ax_colorbar)
        cbar.set_label(label, labelpad = 15)

        # Removing x and y labels
        for ax in [ax_top_plot, ax_bottom_plot_1, ax_bottom_plot_2, ax_bottom_plot_3]:
            ax.set_xticks([])
            ax.set_yticks([])

        # Adjust spacing
        plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, wspace = 0.01, hspace = 0.15)

        # Complete figure label
        fig_name = label + " (D" + str(index_day) + ")"

        return [fig, fig_name]

    def compute_plots_regression(self, y_pred: np.array, y_true: np.array, index_day : int):
        r"""Computes pixel-wise regression metrics, i.e. returns a tensor of shape (x, y)"""

        # Retrieving dimensions for ease of use
        t, x, y = y_true.shape

        # Preprocessing, i.e (t, x, y) to (t, x * y)
        y_true = y_true.reshape(t, -1)
        y_pred = y_pred.reshape(t, -1)

        # Definition of the metrics
        metrics_regression = [MeanSquaredError(num_outputs = x * y),
                              RootMeanSquaredErrorPerPixel(),
                              PercentageOfBiasPerPixel(),
                              PearsonCorrCoef(num_outputs = x * y)]

        metrics_names = ["Mean Squared Error",
                         "Root Mean Squared Error",
                         "Percentage of Bias",
                         "Pearson Correlation Coefficient"]

        metrics_range = [(0,     0.25),
                         (0,     0.50),
                         (-0.5,   0.5),
                         (-1,       1)]

        # Stores all the scores plots, i.e. results of the metrics per pixel
        scores = list()

        for m, m_name, m_range in zip(metrics_regression, metrics_names, metrics_range):

            # Computing score
            score = m(y_pred, y_true).reshape(x, y)

            # Masking the land and non-observed region, i.e. NaNs are simply white when plotted
            score[self.mask[:-2, :-2] == 0] = np.nan

            # Adding results, i.e. fig and name
            scores.append(self.make_plots(score, index_day, m_name, "PuOr", m_range))

        return scores

    def compute_plots_classification(self, y_pred: np.array, y_true: np.array, index_day : int):
        r"""Computes pixel-wise regression metrics, i.e. returns a tensor of shape (x, y)"""

        # Retrieving dimensions for ease of use
        t, x, y = y_true.shape

        # Preprocessing, i.e (t, x, y) to (t, x * y) then (x * y, t)
        y_true = np.swapaxes(y_true.reshape(t, -1), 0, 1)
        y_pred = np.swapaxes(y_pred.reshape(t, -1), 0, 1)

        # Definition of the metrics
        metrics_classification = [BinaryAccuracy(multidim_average  = 'samplewise'),
                                  BinaryPrecision(multidim_average = 'samplewise'),
                                  BinaryRecall(multidim_average    = 'samplewise')]

        metrics_names = ["Accuracy",
                         "Precision",
                         "Recall"]

        metrics_range = [(0, 1),
                         (0, 1),
                         (0, 1)]

        # Stores all the scores, i.e. results of the metrics
        scores = list()

        for m, m_name, m_range in zip(metrics_classification, metrics_names, metrics_range):

            # Computing score
            score = m(y_pred, y_true).reshape(x, y)

            # Masking the land and non-observed region, i.e. NaNs are simply white when plotted
            score[self.mask[:-2, :-2] == 0] = np.nan

            # Adding results, i.e. fig and name
            scores.append(self.make_plots(score, index_day, m_name, "RdYlBu", m_range))

        return scores

    def compute_plots_classification_ROCAUC(self, y_pred: np.array, y_true: np.array, index_day : int):
        r"""Computes the ROCAUC metric and returns a plot of it"""

        # Preprocessing, i.e (t, x, y) to (t, x * y) then (x * y, t)
        y_true = np.swapaxes(y_true, 1, 2).reshape(-1, 2).type(torch.int64)
        y_pred = np.swapaxes(y_pred, 1, 2).reshape(-1, 2)

        # Loading tools
        roc_curve = BinaryROC()
        auroc     = BinaryAUROC()

        # Computing false positive rates, true positive rates and thresholds
        fpr, tpr, thresholds = roc_curve(y_pred, y_true)
        auc_score            = auroc(y_pred, y_true)

        # Plotting the results
        fig = plt.figure(figsize = (7, 7))
        plt.plot(fpr, tpr, label = f'AUC = {auc_score:.2f}', color = 'red', alpha = 0.6)
        plt.plot([0, 1], [0, 1], linestyle = '--', color = 'gray', label = 'Random')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.grid()
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.annotate(f'AUC = {auc_score:.2f}',
                     xy = (0.95, 0.05),
                     xycoords = 'axes fraction',
                     ha = 'right',
                     va = 'bottom',
                     fontsize = 10,
                     bbox = dict(boxstyle = 'round', facecolor = 'white', alpha = 0.7))

        # Complete fig name
        fig_name = "Area Under The Curve (D" + str(index_day) + ")"

        return [[fig, fig_name]]
