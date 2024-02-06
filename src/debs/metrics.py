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

# Pytorch
import torch

# Torch metrics (from Pytorch Lightning)
from torchmetrics.regression     import MeanSquaredError, PearsonCorrCoef, R2Score
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryMatthewsCorrCoef, BinaryROC, BinaryAUROC


# ----------------------
#        Metrics
# ----------------------
# Definition of new metrics (using lambdas) which do not exist in Pytorch Lightning Metrics
#
def PercentageOfBias(y_pred : np.array = None, y_true : np.array = None):
    r"""Used to compute the percentage of bias"""
    return lambda y_true, y_pred  : np.nanmean((y_true - y_pred) / np.abs(y_true))

def RootMeanSquaredError(y_pred : np.array = None, y_true : np.array = None):
    r"""Used to compute the root mean squared error"""
    return lambda y_true, y_pred  : np.sqrt(np.nanmean((y_true - y_pred) ** 2))

def PercentageOfBiasPerPixel(y_pred : np.array = None, y_true : np.array = None):
    r"""Used to compute the percentage of bias per pixel (used for plots)"""
    return lambda y_true, y_pred  : np.nanmean((y_true - y_pred) / np.abs(y_true), axis = 0)

def RootMeanSquaredErrorPerPixel(y_pred : np.array = None, y_true : np.array = None):
    r"""Used to compute the root mean squared error per pixel (used for plots)"""
    return lambda y_true, y_pred  : np.sqrt(np.nanmean((y_true - y_pred) ** 2, axis = 0))


class BlackSea_Metrics():
    r"""A tool to compute a large variety of metrics (scalar or visual) to assess the quality of a model."""

    def __init__(self, mode: str, mask : np.array, treshold : float, number_of_batches : int):
        r"""Initialization of the metrics computation tool"""

        # Storing information
        self.mask = mask
        self.mode = mode
        self.number_of_batches = number_of_batches
        self.treshold_normalized_oxygen = treshold

        # Used to store results and plots
        self.scores, self.scores_names, self.plots = None, None, list()

        # Definition of the metrics
        self.metrics_regression     = [MeanSquaredError(), RootMeanSquaredError(), R2Score(), PearsonCorrCoef(), PercentageOfBias()]
        self.metrics_classification = [BinaryAccuracy(), BinaryPrecision(), BinaryRecall(), BinaryMatthewsCorrCoef()]

        # Definition of the names of each metric
        self.metrics_regression_names     = ["Mean Squared Error", "Root Mean Squared Error", "R2 Score", "Pearson Correlation Coefficient", "Percentage of Bias"]
        self.metrics_classification_names = ["Accuracy",  "Precision",  "Recall", "Matthews Correlation Coefficient"]

    def get_names_metrics(self):
        r"""Retreives the name of all the metrics (scalar)"""
        return      ["Area Under The Curve"] + self.metrics_classification_names if self.mode == "classification" else \
               self.metrics_regression_names + self.metrics_classification_names

    def get_names_plots(self):
        r"""Retreives the name of all the metrics (visual)"""
        return      ["Area Under The Curve"] + ["Accuracy", "Precision", "Recall"] if self.mode == "classification" else \
                    ["Mean Squared Error", "Root Mean Squared Error", "Percentage of Bias", "Pearson Correlation Coefficient"] + ["Accuracy", "Precision", "Recall"]

    def get_results(self):
        r"""Retreives the results of all the metrics"""
        return self.scores, self.get_names_metrics()

    def get_plots(self):
        r"""Retreives the plots of all the metrics"""
        return self.plots, self.get_names_plots()

    def compute_metrics(self, y_pred: np.array, y_true: np.array):
        r"""Computes each metric (scalar) for each individual forecasted day, i.e. returns a tensor of shape [forecasted days, metrics]"""

        # Retrieving dimensions for ease of comprehension
        batch_size, days, = y_true.shape[0], y_true.shape[1]

        # Stores results for each days, convert to numpy and average over number of batches (everything is summed so at the end, we will have the true average)
        scores_temporary = np.array([self.compute_metrics_(y_pred[:, i], y_true[:, i]) for i in range(days)]) / self.number_of_batches

        # Adding the results to previous one or initialize it
        self.scores = np.sum([self.scores, scores_temporary], axis = 0) if isinstance(self.scores, (np.ndarray, np.generic)) else \
                      np.array(scores_temporary)

    def compute_metrics_(self, y_pred_per_day: np.array, y_true_per_day: np.array):
        r"""Computes the metrics (scalar) for a given day and returns them as a list of values"""

        # Stores all the scores
        results = list()

        # ------- CLASSIFICATION (ROCAUC) -------
        if self.mode == "classification":

            # Retrieving values in the sea, swapping axis (t, c, x, y) to (c, t, x, y) and flattening (c, t * x * y)
            y_true_per_day = np.swapaxes(y_true_per_day[:, :, self.mask[:-2, :-2] == 1], 0, 1).reshape(2, -1)
            y_pred_per_day = np.swapaxes(y_pred_per_day[:, :, self.mask[:-2, :-2] == 1], 0, 1).reshape(2, -1)

            # Used th compute the area under the curve
            toolAUC = BinaryAUROC()

            # Computations
            results += [toolAUC(y_pred_per_day, y_true_per_day).item()]

            # Transforming the problem to non-probabilistic
            y_true_per_day = np.argmax(y_true_per_day, axis = 0)
            y_pred_per_day = np.argmax(y_pred_per_day, axis = 0)

        # ------- REGRESSION -------
        if self.mode == "regression":

            # Retrieving values in the sea and flattening (t * x * y)
            y_true_per_day = y_true_per_day[:, self.mask[:-2, :-2] == 1].reshape(-1)
            y_pred_per_day = y_pred_per_day[:, self.mask[:-2, :-2] == 1].reshape(-1)

            # Computations
            results += [metric(y_pred_per_day, y_true_per_day).item() for metric in self.metrics_regression]

            # Transforming problem to classification
            y_true_per_day = (y_true_per_day < self.treshold_normalized_oxygen) * 1
            y_pred_per_day = (y_pred_per_day < self.treshold_normalized_oxygen) * 1

        # ------- CLASSIFICATION (ACC, PRE, ...) -------
        return results + [metric(y_pred_per_day, y_true_per_day).item() for metric in self.metrics_classification]

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

        return fig

    def compute_plots(self, y_pred: np.array, y_true: np.array):
        r"""Computes each metric per pixel for each individual forecasted day and show the results on a plot"""

        # Retrieving dimensions for ease of comprehension
        batch_size, days, = y_true.shape[0], y_true.shape[1]

        # Looping over each day
        for i in range(days):

            # Retrieving corresponding day
            y_true_per_day = y_true[:, i]
            y_pred_per_day = y_pred[:, i]

            # ------- CLASSIFICATION (ROCAUC) -------
            if self.mode == "classification":

                # Drawing plot
                self.plots += self.compute_plots_classification_ROCAUC(y_pred_per_day[:, :, self.mask[:-2, :-2] == 1], y_true_per_day[:, :, self.mask[:-2, :-2] == 1], i)

                # Transforming problem to non-probabilistic
                y_true_per_day = np.argmax(y_true_per_day, axis = 1)
                y_pred_per_day = np.argmax(y_pred_per_day, axis = 1)

            # ------- REGRESSION -------
            if self.mode == "regression":

                # Drawing plots
                self.plots += self.compute_plots_regression(y_pred_per_day, y_true_per_day, i)

                # Transforming problem to classification
                y_pred_per_day = (y_pred_per_day < self.treshold_normalized_oxygen) * 1
                y_true_per_day = (y_true_per_day < self.treshold_normalized_oxygen) * 1

            # ------- CLASSIFICATION (ACC, PRE, ...) -------
            self.plots += self.compute_plots_classification(y_pred_per_day, y_true_per_day, i)

    def compute_plots_regression(self, y_pred: np.array, y_true: np.array, index_day : int):
        r"""Computes each metric per pixel and show the results on a plot"""

        # Retrieving dimensions (Ease of comprehension)
        t, x, y = y_true.shape

        # Preprocessing, i.e (t, x, y) to (t, x * y)
        y_true = y_true.reshape(t, -1)
        y_pred = y_pred.reshape(t, -1)

        # Definition of the pixelwise metrics (Pytorch Lightning Metrics and customs)
        metrics_regression = [MeanSquaredError(num_outputs = x * y), RootMeanSquaredErrorPerPixel(), PercentageOfBiasPerPixel(), PearsonCorrCoef(num_outputs = x * y)]

        # Labels for the plots
        metrics_names = ["Mean Squared Error",
                         "Root Mean Squared Error",
                         "Percentage of Bias",
                         "Pearson Correlation Coefficient"]

        # Definition of the range for the colorbar
        metrics_range = [(0, 0.25), (0, 0.50), (-0.5, 0.5), (-1, 1)]

        # Stores all the plots
        plots = list()

        for metric, name, limits in zip(metrics_regression, metrics_names, metrics_range):

            # Computing score
            score = metric(y_pred, y_true).reshape(x, y)

            # Masking the land and non-observed region, i.e. NaNs are white when plotted so thats the best !
            score[self.mask[:-2, :-2] == 0] = np.nan

            # Adding results, i.e. fig and name
            plots.append(self.make_plots(score, index_day, name, "PuOr", limits))

        return plots

    def compute_plots_classification(self, y_pred: np.array, y_true: np.array, index_day : int):
        r"""Computes each metric per pixel and show the results on a plot"""

        # Retrieving dimensions for ease of use
        t, x, y = y_true.shape

        # Preprocessing, i.e (t, x, y) to (t, x * y) then finally (x * y, t)
        y_true = np.swapaxes(y_true.reshape(t, -1), 0, 1)
        y_pred = np.swapaxes(y_pred.reshape(t, -1), 0, 1)

        # Definition of the pixelwise metrics (Pytorch Lightning Metrics and customs)
        metrics_classification = [BinaryAccuracy(multidim_average  = 'samplewise'),
                                  BinaryPrecision(multidim_average = 'samplewise'),
                                  BinaryRecall(multidim_average    = 'samplewise')]

        metrics_names = ["Accuracy",
                         "Precision",
                         "Recall"]

        # Definition of the range for the colorbar
        metrics_range = [(0, 1), (0, 1), (0, 1)]

        # Stores all the plots
        scores = list()

        for metric, name, limits in zip(metrics_classification, metrics_names, metrics_range):

            # Computing score
            score = metric(y_pred, y_true).reshape(x, y)

            # Masking the land and non-observed region, i.e. NaNs are white when plotted so thats the best !
            score[self.mask[:-2, :-2] == 0] = np.nan

            # Adding results, i.e. fig and name
            scores.append(self.make_plots(score, index_day, name, "RdYlBu", limits))

        return scores

    def compute_plots_classification_ROCAUC(self, y_pred: np.array, y_true: np.array, index_day : int):
        r"""Computes the ROC metric and returns a plot of it with the AUC value"""

        # Preprocessing, i.e (t, x, y) to (t, x * y) then (x * y, t)
        y_true = np.swapaxes(y_true, 1, 2).reshape(-1, 2).type(torch.int64)
        y_pred = np.swapaxes(y_pred, 1, 2).reshape(-1, 2)

        # Loading Pytorch Metrics Tools
        roc_curve = BinaryROC()
        auroc     = BinaryAUROC()

        # Computing false positive rates, true positive rates and thresholds
        fpr, tpr, thresholds = roc_curve(y_pred, y_true)
        auc_score            = auroc(y_pred, y_true)

        # Plotting the ROC curve
        fig = plt.figure(figsize = (7, 7))
        plt.plot(fpr, tpr, label = f'AUC = {auc_score:.2f}', color = 'red', alpha = 0.6)
        plt.plot([0, 1], [0, 1], linestyle = '--', color = 'gray', label = 'Random')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.grid()
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

        # Adding the Area Under The Curve value
        plt.annotate(f'AUC = {auc_score:.2f}',
                     xy = (0.95, 0.05),
                     xycoords = 'axes fraction',
                     ha = 'right',
                     va = 'bottom',
                     fontsize = 10,
                     bbox = dict(boxstyle = 'round', facecolor = 'white', alpha = 0.7))

        # Complete fig name
        fig_name = "Area Under The Curve (D" + str(index_day) + ")"

        return [fig]
