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
from torchmetrics.classification import BinaryROC, BinaryAUROC, BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryMatthewsCorrCoef, MulticlassAUROC,BinaryF1Score


# ----------------------
#        Metrics
# ----------------------
# Definition of new metrics that do not exist in Pytorch Lightning Metrics
#
def PercentageOfBias(y_pred : np.array = None, y_true : np.array = None):
    r"""Used to compute the percentage of bias"""
    return lambda y_true, y_pred  : torch.nanmean((y_true - y_pred) / torch.abs(y_true), dim = 0)

def RootMeanSquaredError(y_pred : np.array = None, y_true : np.array = None):
    r"""Used to compute the root mean squared error"""
    return lambda y_true, y_pred  : torch.sqrt(torch.nanmean((y_true - y_pred) ** 2, dim = 0))

def PercentageOfBiasPerPixel(y_pred : np.array = None, y_true : np.array = None):
    r"""Used to compute the percentage of bias per pixel (used for plots)"""
    return lambda y_true, y_pred  : np.nanmean((y_true - y_pred) / np.abs(y_true), axis = 0)

def RootMeanSquaredErrorPerPixel(y_pred : np.array = None, y_true : np.array = None):
    r"""Used to compute the root mean squared error per pixel (used for plots)"""
    return lambda y_true, y_pred  : np.sqrt(np.nanmean((y_true - y_pred) ** 2, axis = 0))


class BlackSea_Metrics():
    r"""A tool to compute a large variety of metrics (scalar or visual) to assess the quality of a model."""

    def __init__(self, mode: str, mask : np.array, mask_complete : np.array, treshold : float, number_of_samples : int):
        r"""Initialization of the metrics computation tool"""

        # Storing information
        self.mask = mask
        self.mode = mode
        self.mask_complete = mask_complete
        self.number_of_samples = number_of_samples
        self.treshold_normalized_oxygen = treshold

        # Creation of a mask for plots
        self.mask_plots = np.flipud(self.mask == 1)

        # Used to store results and plots
        self.scores, self.scores_names, self.plots = None, None, list()

        # Definition of the names of each metric
        self.metrics_regression_names     = ["Percentage of Bias", "Mean Squared Error", "Root Mean Squared Error", "R2 Score", "Pearson Correlation Coefficient"]
        self.metrics_classification_names = ["Accuracy",  "Precision",  "Recall", "F1-Score"]

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

        # Selecting the mean, i.e. metrics are computed with the mean value predicted
        if self.mode == "regression":
            y_pred = y_pred[:, :, 0]
            y_true = y_true[:, :, 0]

        # Retrieving dimensions for ease of comprehension
        batch_size, days, = y_true.shape[0], y_true.shape[1]

        # Stores results for each days, convert to numpy and average over number of batches (everything is summed so at the end, we will have the true average)
        scores_temporary = np.array([self.compute_metrics_(y_pred[:, i], y_true[:, i]) for i in range(days)]) / self.number_of_samples

        # Adding the results to previous one or initialize it
        self.scores = np.sum([self.scores, scores_temporary], axis = 0) if isinstance(self.scores, (np.ndarray, np.generic)) else \
                      np.array(scores_temporary)

    def compute_metrics_(self, y_pred_per_day: np.array, y_true_per_day: np.array):
        r"""Computes the metrics (scalar) for a given day and returns them as a list of values"""

        # Stores all the scores
        results = list()

        # ------- CLASSIFICATION (ROCAUC) -------
        if self.mode == "classification":

            # Retrieving dimensions (Ease of comprehension)
            t, c, x, y = y_true_per_day.shape

            # Retrieving values in the sea, swapping axis (t, c, x, y) to (c, t, x, y) and flattening (c, t * x * y)
            y_true_per_day = y_true_per_day[:, :, self.mask[:, :] == 1].reshape(t, c, -1)
            y_pred_per_day = y_pred_per_day[:, :, self.mask[:, :] == 1].reshape(t, c, -1)

            # Used th compute the area under the curve
            toolAUC = MulticlassAUROC(num_classes = c)

            # Transforming the problem to non-probabilistic
            y_true_per_day = np.argmax(y_true_per_day, axis = 1)

            # Computations
            results += [toolAUC(y_pred_per_day, y_true_per_day).item() * self.number_of_samples]

            # Transforming the problem to non-probabilistic
            y_pred_per_day = np.argmax(y_pred_per_day, axis = 1)

        # ------- REGRESSION -------
        if self.mode == "regression":

            # Retrieving dimensions (Ease of comprehension)
            t = y_true_per_day.shape[0]

            # Metrics to perform regression (Computed on each sample, everything is then summed and averaged over the number of samples afterwards)
            metrics_regression = [PercentageOfBias(),
                                  MeanSquaredError(num_outputs = t),
                                  RootMeanSquaredError(),
                                  R2Score(num_outputs = t, multioutput = 'raw_values'),
                                  PearsonCorrCoef( num_outputs = t)]

            # Dataset for precision and recall (only computed on region where hypoxia can occur)
            y_true_per_day_hyp = torch.swapaxes(y_true_per_day[:, self.mask_complete[:, :] >= 1], 0, 1)
            y_pred_per_day_hyp = torch.swapaxes(y_pred_per_day[:, self.mask_complete[:, :] >= 1], 0, 1)

            # Swapping axes (t, x * y) to (x * y, t)
            y_true_per_day = torch.swapaxes(y_true_per_day[:, self.mask[:, :] == 1], 0, 1)
            y_pred_per_day = torch.swapaxes(y_pred_per_day[:, self.mask[:, :] == 1], 0, 1)

            # Computations
            results += [torch.sum(metric(y_pred_per_day, y_true_per_day)).item() for metric in metrics_regression]

            # Transforming problem to classification
            y_true_per_day = (y_true_per_day < self.treshold_normalized_oxygen) * 1
            y_pred_per_day = (y_pred_per_day < self.treshold_normalized_oxygen) * 1

            y_true_per_day_hyp = (y_true_per_day_hyp < self.treshold_normalized_oxygen) * 1
            y_pred_per_day_hyp = (y_pred_per_day_hyp < self.treshold_normalized_oxygen) * 1

            # Swapping axes (x * y, t) to (t, x * y)
            y_true_per_day = torch.swapaxes(y_true_per_day, 0, 1)
            y_pred_per_day = torch.swapaxes(y_pred_per_day, 0, 1)

            y_true_per_day_hyp = torch.swapaxes(y_true_per_day_hyp, 0, 1)
            y_pred_per_day_hyp = torch.swapaxes(y_pred_per_day_hyp, 0, 1)

        # ------- CLASSIFICATION (ACC, PRE, ...) -------
        #
        # Defining metrics
        acc = BinaryAccuracy( multidim_average = 'samplewise')
        pre = BinaryPrecision(multidim_average = 'samplewise')
        rec = BinaryRecall(   multidim_average = 'samplewise')
        f1  = BinaryF1Score(  multidim_average = 'samplewise')

        # Metrics to perform classification (Computed on each sample, everything is then summed and averaged over the number of samples afterwards)
        results = results + [torch.sum(acc(y_pred_per_day,     y_true_per_day)).item()]
        results = results + [torch.sum(pre(y_pred_per_day_hyp, y_true_per_day_hyp)).item()]
        results = results + [torch.sum(rec(y_pred_per_day_hyp, y_true_per_day_hyp)).item()]
        results = results + [torch.sum( f1(y_pred_per_day,     y_true_per_day)).item()]

        return results

    def make_plots(self, score : np.array, index_day : int, label : str, cmap : str, vminmax : tuple):
        r"""Creates a custom plot for each metric"""

        # Flipping vertically to show correctly Black Sea (for you my loving oceanographer <3)
        score = score.numpy() if isinstance(score, torch.Tensor) else score

        # Working (self.mask does not work)
        mask_current             = ~(self.mask_complete[:, :] >= 0)
        score[mask_current == 1] = np.nan

        # Hides all the regions that are not relevant for this metric
        if "Precision" in label or "Recall" in label:
            score[self.mask_complete[:, :] == 0] = np.nan

        # Fplipping vertically
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
        ax_top_plot.imshow(self.mask_plots[20:, :], cmap = "grey", alpha = 0.1, aspect = '0.83')


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
                                      aspect = '0.55')
        ax_bottom_plot_1.imshow(self.mask_plots[40:110, 175:275], cmap = "grey", alpha = 0.1, aspect = '0.55')

        # Plotting (3) - Focusing on top bottom region
        ax_bottom_plot_2 = fig.add_subplot(gs[1, 1])
        im3 = ax_bottom_plot_2.imshow(score[62:100, 250:470],
                                      cmap   = cmap,
                                      vmin   = vminmax[0],
                                      vmax   = vminmax[1],
                                      aspect = '4.5')
        ax_bottom_plot_2.imshow(self.mask_plots[82:100, 250:470], cmap = "grey", alpha = 0.1, aspect = '4.5')


        # Plotting (4) - Focusing on bottom region
        ax_bottom_plot_3 = fig.add_subplot(gs[2, :-1])
        im4 = ax_bottom_plot_3.imshow(score[200:, :500],
                                      cmap   = cmap,
                                      vmin   = vminmax[0],
                                      vmax   = vminmax[1],
                                      aspect = '3')
        ax_bottom_plot_3.imshow(self.mask_plots[220:, :500], cmap = "grey", alpha = 0.1, aspect = '3')

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
                self.plots += self.compute_plots_classification_ROCAUC(y_pred_per_day[:, :, self.mask[:, :] == 1], y_true_per_day[:, :, self.mask[:, :] == 1], i)

                # Transforming problem to non-probabilistic
                y_true_per_day = np.argmax(y_true_per_day, axis = 1)
                y_pred_per_day = np.argmax(y_pred_per_day, axis = 1)

            # ------- REGRESSION -------
            if self.mode == "regression":

                # Extracting the mean value
                y_pred_per_day = y_pred_per_day[:, 0]
                y_true_per_day = y_true_per_day[:, 0]

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
        metrics_names = ["Mean Squared Error $[(mmol/m^3)^2]$",
                         "Root Mean Squared Error $[mmol/m^3$]",
                         "Percentage of Bias [%]",
                         "Pearson Correlation Coefficient [-]"]

        # Definition of the range for the colorbar
        metrics_ranges = [(0, 0.1), (0, 0.1), (-100, 100), (-1, 1)]

        # Colors for the plots (need to have sequential colors and diverging )
        metrics_colors = ["coolwarm", "coolwarm", "RdBu", "RdBu"]

        # Stores all the plots
        plots = list()

        for metric, name, limits, color in zip(metrics_regression, metrics_names, metrics_ranges, metrics_colors):

            # Computing score
            score = metric(y_pred, y_true).reshape(x, y)

            # Adding results, i.e. fig and name (multipliy by 100 for percentage)
            plots.append(self.make_plots(score * limits[1], index_day, name, color, limits))

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

        metrics_names = ["Accuracy [%]",
                         "Precision [%]",
                         "Recall [%]"]

        # Definition of the range for the colorbar
        metrics_range = [(0, 100), (0, 100), (0, 100)]

        # Stores all the plots
        scores = list()

        for metric, name, limits in zip(metrics_classification, metrics_names, metrics_range):

            # Computing score
            score = metric(y_pred, y_true).reshape(x, y)

            # Adding results, i.e. fig and name
            scores.append(self.make_plots(score * 100, index_day, name, "Spectral", limits))

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

    def compute_plots_comparison_classification(self, y_pred : torch.tensor, y_true : torch.tensor):
        """Plot and compare two tensors"""

        def convert_prob_to_classification(tensor : torch.tensor):
            """Convert probabilities to classification"""
            if len(tensor.shape) == 5:
                mask = tensor[0, 0, 0] == -1
                tensor = torch.argmax(tensor, dim=2)
            else:
                mask = (tensor[0, 0] == -1) * 1
            return tensor, mask

        # Determining the type of problem
        prob_type = len(y_pred.shape)

        # Defining color map
        cmap = "viridis" if prob_type == 4 else "viridis"

        # Convert probabilities to classification
        y_pred, _         = convert_prob_to_classification(y_pred)
        y_true, mask_true = convert_prob_to_classification(y_true)

        # Hiding the land
        y_pred[:, :, mask_true == 1] = np.nan if prob_type == 4 else -1
        y_true[:, :, mask_true == 1] = np.nan if prob_type == 4 else -1

        # Flipping vertically (ease of comprehension)
        y_pred = torch.flip(y_pred, dims = (2,))
        y_true = torch.flip(y_true, dims = (2,))

        # Plotting the results
        fig, axes = plt.subplots(2, 1, figsize = (20, 10))

        # Plot Prediction
        im1 = axes[0].imshow(y_pred[0, 0], cmap=cmap, vmin = 0 if prob_type == 4 else -1, vmax = 1)  # Set vmin and vmax
        axes[0].set_ylabel('Prediction')
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        # Add colorbar to the right of the Prediction plot
        cbar1 = fig.colorbar(im1, ax=axes[0], fraction = 0.025, pad = 0.04)

        # Plot Ground Truth
        im2 = axes[1].imshow(y_true[0, 0], cmap=cmap, vmin = 0 if prob_type == 4 else -1, vmax = 1)  # Set vmin and vmax
        axes[1].set_ylabel('Ground Truth')
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        # Add colorbar to the right of the Ground Truth plot
        cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.025, pad=0.04)

        # Adjust layout
        plt.tight_layout()
        plt.show()

        return fig

    def compute_plots_comparison_regression(self, y_pred : torch.tensor, y_true : torch.tensor):
            """Plot and compare two tensors"""

            # Defining color map
            cmap = "viridis"

            # Extracting the mean and standard deviation
            y_true      =           y_true[:, 0, 0]
            y_pred_mean =           y_pred[:, 0, 0]
            y_pred_std  = torch.exp(y_pred[:, 0, 1]/2)

            # Hiding the land
            y_pred_mean[:, self.mask == 0] = np.nan
            y_pred_std[ :, self.mask == 0] = np.nan
            y_true[     :, self.mask == 0] = np.nan

            # Flipping vertically (ease of comprehension)
            y_pred_mean = torch.flipud(y_pred_mean[0])
            y_pred_std  = torch.flipud(y_pred_std[0])
            y_true      = torch.flipud(y_true[0]) if isinstance(y_true, torch.Tensor) else np.flipud(y_true[0])

            # Plotting the results
            fig, axes = plt.subplots(3, 1, figsize = (12, 12))

            # Plot Standard Devaition
            im1 = axes[0].imshow(y_pred_std, cmap=cmap, vmin = 0, vmax = 1)
            axes[0].imshow(self.mask_plots, cmap = "grey", alpha=0.1)
            axes[0].set_ylabel('Uncertainty (STD)')
            axes[0].set_xticks([])
            axes[0].set_yticks([])
            cbar1 = fig.colorbar(im1, ax=axes[0], fraction = 0.025, pad = 0.04)

            # Plot Mean Prediction
            im2 = axes[1].imshow(y_pred_mean, cmap=cmap, vmin = 0, vmax = 1)
            axes[1].imshow(self.mask_plots, cmap = "grey", alpha=0.1)
            axes[1].set_ylabel('Prediction (M)')
            axes[1].set_xticks([])
            axes[1].set_yticks([])
            cbar2 = fig.colorbar(im2, ax=axes[1], fraction = 0.025, pad = 0.04)

            # Plot Ground Truth
            im3 = axes[2].imshow(y_true, cmap=cmap, vmin = 0, vmax = 1)
            axes[2].imshow(self.mask_plots, cmap = "grey", alpha=0.1)
            axes[2].set_ylabel('Ground Truth')
            axes[2].set_xticks([])
            axes[2].set_yticks([])

            # Add colorbar to the right of the Ground Truth plot
            cbar3 = fig.colorbar(im3, ax=axes[2], fraction=0.025, pad=0.04)

            # Adjust layout
            plt.tight_layout()
            plt.show()

            return fig

    def compute_plot_ROCAUC_global(self, y_pred: torch.tensor, y_true: torch.tensor, normalized_threshold: float):
        """Used to plot the ROC curve for different values of the threshold"""

        # Exctrating the first day
        y_pred = y_pred[:, 0]
        y_true = y_true[:, 0]

        # Extracting the mean
        y_pred = y_pred[:, 0]
        y_true = y_true[:, 0]

        # Extracting only the observed region
        y_pred = y_pred[:, self.mask == 1]
        y_true = y_true[:, self.mask == 1]

        # Retrieving dimensions (Ease of comprehension)
        samples, xy = y_pred.shape

        # Flatenning everything
        y_pred = y_pred.view(samples, -1)
        y_true = y_true.view(samples, -1)

        # Simple linspace between 0 and 1
        threshold = torch.linspace(0, 1, 100)

        # Stores the ROC curve for each threshold
        false_positive, true_positive = list(), list()

        # Looping over all the possible tresholds
        for t in threshold:

            # Conversion to binary classification
            y_pred_t = (y_pred < t)                    * 1.
            y_true_t = (y_true < normalized_threshold) * 1

            # Initialize ROC metric from torchmetrics for binary classification task
            ROC_tool = BinaryROC(thresholds = [0.5])

            # Compute ROC curve
            fp, tp, _  = ROC_tool(y_pred_t, y_true_t)

            # Appending the results
            false_positive.append(fp), true_positive.append(tp)

        # Conversion to tensors
        false_positive = torch.as_tensor(false_positive)
        true_positive  = torch.as_tensor(true_positive)

        # Sort the false positive in ascending order and sort the true positive accordingly
        sorted_indices = torch.argsort(false_positive)
        false_positive = false_positive[sorted_indices]
        true_positive  = true_positive[sorted_indices]

        # Computing the AREA under the curve
        area = torch.trapz(true_positive, false_positive)

        # Plotting the results
        fig = plt.figure(figsize = (10, 10))
        plt.plot(false_positive, true_positive, color='darkorange', lw=2, label = f'Area = {area:.2f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc='lower right')
        plt.grid(alpha = 0.5)
        plt.show()

        # Returning
        return false_positive, true_positive, area, fig

    def compute_plot_ROCAUC_local(self, y_pred: torch.tensor, y_true: torch.tensor, normalized_threshold: float):
        """Computes and plots the ROC curve for different values of the threshold"""

        # Extracting the current day
        y_pred = y_pred[:, 0]
        y_true = y_true[:, 0]

        # Extracting the mean
        y_pred = y_pred[:, 0]
        y_true = y_true[:, 0]

        # Extracting shape informaiton
        t, x, y    = y_pred.shape

        # Simple linspace between 0 and 1
        threshold = torch.linspace(0, 1, 1)

        # Reshaping the data for easier computation
        y_pred = y_pred.reshape(t, x * y).permute(1, 0)
        y_true = y_true.reshape(t, x * y).permute(1, 0)

        # Initialize ROC metric from torchmetrics for binary classification task
        ROC_tool = BinaryROC(thresholds=[0.5])

        # Compute ROC curve for each threshold
        auc = []
        for i in range(x * y):
            false_positive_current, true_positive_current = [], []
            for t in threshold:

                # Convert to binary classification based on threshold
                y_pred_t = (y_pred[i] < t) * 1.
                y_true_t = (y_true[i] < normalized_threshold) * 1

                # Compute ROC curve using torchmetrics
                fp, tp, _ = ROC_tool(y_pred_t, y_true_t)
                false_positive_current.append(fp)
                true_positive_current.append(tp)

            # Convert lists to tensors
            false_positive_current = torch.as_tensor(false_positive_current)
            true_positive_current = torch.as_tensor(true_positive_current)

            # Compute area under the curve (AUC)
            auc.append(torch.trapz(true_positive_current, false_positive_current))

        # Reshape AUC values to match the original data dimensions
        auc = torch.as_tensor(auc).reshape(x, y)

        # Conversion to numpy
        auc = auc.numpy()

        # Masking the land
        mask_current             = ~(self.mask_complete[:, :] >= 0)
        auc[mask_current == 1] = np.nan

        # Flipping vertically
        auc = np.flipud(auc)

        # Plot the results
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(auc, cmap="viridis")
        plt.colorbar()
        plt.show()

        # Return the figure
        return fig
