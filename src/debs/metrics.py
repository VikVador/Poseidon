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
import wandb
import numpy as np
import matplotlib.pyplot as plt

# Pytorch
import torch

def metrics(ground_truth: torch.Tensor, forecasts: torch.Tensor, mask: torch.Tensor, treshold: float):
    """Used to compute different visualizations"""

    def plot_forecasts(ground_truth: torch.Tensor, forecasts: torch.Tensor, mask: torch.Tensor, day: int = 0):
        """Plotting the forecasts against the ground truth"""

        # Plotting the forecasts
        fig, ax = plt.subplots(12, 4, figsize = (30, 50))

        # List of months
        months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

        # Looping over each first day of the month
        for i in range(12):

            # Extracting the region of interest
            gt = ground_truth[i, day, 25:125, 70:270]

            # Masking the ground truth
            gt[mask[0, 25:125, 70:270] == 0] = np.nan

            # Extracting the minimum and maximum values
            vmin, vmax = np.nanmin(gt), np.nanmax(gt)

            # Plotting the ground truth
            ax[i, 0].imshow(gt, label = "Ground Truth")

            # Removing the tickz
            ax[i, 0].set_xticks([])
            ax[i, 0].set_yticks([])

            # Adding the title
            ax[i, 0].set_ylabel(months[i])

            # Plotting the forecasts
            for j in range(3):

                # Extracting the forecast
                fc = forecasts[i, day, j, 25:125, 70:270]

                # Masking the forecast
                fc[mask[0, 25:125, 70:270] == 0] = np.nan

                # Plotting the forecast
                ax[i, j + 1].imshow(fc, label = f"Forecast {j + 1}", vmin = vmin, vmax = vmax)
                ax[i, j + 1].set_xticks([])
                ax[i, j + 1].set_yticks([])

        # Tight layout
        plt.tight_layout()
        return fig

    def plot_hypoxia(ground_truth: torch.Tensor, forecasts: torch.Tensor, mask: torch.Tensor, day: int = 0):
        """Plotting the forecasts against the ground truth"""

        # Plotting the forecasts
        fig, ax = plt.subplots(12, 4, figsize = (30, 50))

        # List of months
        months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

        # Looping over each first day of the month
        for i in range(12):

            # Extracting the region of interest
            gt = ground_truth[i, day, 25:125, 70:270]

            # Masking the ground truth
            gt[mask[0, 25:125, 70:270] == 0] = -1

            # Extracting the minimum and maximum values
            vmin, vmax = np.nanmin(gt), np.nanmax(gt)

            # Plotting the ground truth
            ax[i, 0].imshow(gt, label = "Ground Truth")

            # Removing the tickz
            ax[i, 0].set_xticks([])
            ax[i, 0].set_yticks([])

            # Adding the title
            ax[i, 0].set_ylabel(months[i])

            # Plotting the forecasts
            for j in range(3):

                # Extracting the forecast
                fc = forecasts[i, day, j, 25:125, 70:270]

                # Masking the forecast
                fc[mask[0, 25:125, 70:270] == 0] = -1

                # Plotting the forecast
                ax[i, j + 1].imshow(fc, label = f"Forecast {j + 1}", vmin = vmin, vmax = vmax)
                ax[i, j + 1].set_xticks([])
                ax[i, j + 1].set_yticks([])

        # Tight layout
        plt.tight_layout()
        return fig

    def plot_probability_maps(ground_truth: torch.Tensor, forecasts: torch.Tensor, mask: torch.Tensor, day: int = 0):
        """Plotting the probability maps"""

        # Plotting the forecasts
        fig, ax = plt.subplots(12, 2, figsize = (10, 30))

        # Computing probability map
        prob_map = torch.mean(forecasts, dim = 2)

        # List of months
        months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

        # Looping over each first day of the month
        for i in range(12):

            # Extracting the region of interest
            gt = ground_truth[i, day, 25:125, 70:270]
            pm = prob_map[i, day, 25:125, 70:270]

            # Masking the ground truth
            gt[mask[0, 25:125, 70:270] == 0] = np.nan
            pm[mask[0, 25:125, 70:270] == 0] = np.nan

            # Extracting the minimum and maximum values
            vmin, vmax = np.nanmin(gt), np.nanmax(gt)

            # Plotting the ground truth
            ax[i, 0].imshow(gt, label = "Ground Truth")

            # Adding the colorbar
            fig.colorbar(ax[i, 1].imshow(pm, vmin = 0, vmax = 1, cmap="inferno"), ax = ax[i, 1])

            # Removing the tickz
            ax[i, 0].set_xticks([])
            ax[i, 0].set_yticks([])
            ax[i, 1].set_xticks([])
            ax[i, 1].set_yticks([])

            # Adding the title
            ax[i, 0].set_ylabel(months[i])

        plt.tight_layout()
        return fig

    def compute_precision_recall(ground_truth: torch.Tensor, forecasts: torch.Tensor, mask: torch.Tensor):
        """Computes the recall and precision of the forecasts"""

        # Store results
        results = {
            'precision': [],
            'recall': [],
            'accuracy': []
        }

        # Define thresholds
        thresholds = [i / 10 for i in range(0, 11)]

        # Computing probability map
        prob_map = torch.mean(forecasts, dim = 2)

        # Extracting only relevant information
        ground_truth = ground_truth[:, :, mask[0] == 1]
        prob_map     = prob_map[:, :, mask[0] == 1]

        # Computing metrics
        for threshold in thresholds:

            # Binarize predictions
            binary_prediction = (prob_map >= threshold).float()

            # Calculate TP, FP, TN, FN
            TP = (binary_prediction * ground_truth).sum(dim=(0, 2))
            FP = (binary_prediction * (1 - ground_truth)).sum(dim=(0, 2))
            TN = ((1 - binary_prediction) * (1 - ground_truth)).sum(dim=(0, 2))
            FN = ((1 - binary_prediction) * ground_truth).sum(dim=(0, 2))

            # Compute precision, recall, and accuracy
            precision =       TP / (TP + FP + 1e-8)
            recall =          TP / (TP + FN + 1e-8)
            accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)

            # Adding results to the dictionary
            results['precision'].append(precision)
            results['recall'].append(recall)
            results['accuracy'].append(accuracy)

        # Convert lists to tensors for better handling
        results['precision'] = torch.stack(results['precision'], dim=0)
        results['recall'] = torch.stack(results['recall'], dim=0)
        results['accuracy'] = torch.stack(results['accuracy'], dim=0)

        # Computing Recall vs Precision curve
        rec_pre_0 = plt.figure(figsize=(7, 7))
        plt.plot(results['recall'][:, 0], results['precision'][:, 0], marker='*')
        plt.xlabel('Recall [-]')
        plt.ylabel('Precision [-]')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid()

        rec_pre_1 = plt.figure(figsize=(7, 7))
        plt.plot(results['recall'][:, -1], results['precision'][:,-1], marker='o')
        plt.xlabel('Recall [-]')
        plt.ylabel('Precision [-]')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid()

        # Sending results to WandB
        wandb.log({"Precision-Recall Curve (First Day)": wandb.Image(rec_pre_0),
                   "Precision-Recall Curve (Last Day)":  wandb.Image(rec_pre_1)})

        # Sending singular values and averages
        for i, t in enumerate(thresholds):
            wandb.log({f"Metrics/Precision (First Day, T = {t}))" : results['precision'][i, 0],
                       f"Metrics/Precision (Last Day, T = {t}))"  : results['precision'][i, -1],
                       f"Metrics/Recall (First Day, T = {t}))"    : results['recall'][i, 0],
                       f"Metrics/Recall (Last Day, T = {t}))"     : results['recall'][i, -1],
                       f"Metrics/Accuracy (First Day, T = {t}))"  : results['accuracy'][i, 0],
                       f"Metrics/Accuracy (Last Day, T = {t}))"   : results['accuracy'][i, -1]})

    # Plotting the forecasts
    wandb.log({"Forecast Visualization (First Day)": wandb.Image(plot_forecasts(ground_truth, forecasts, mask, day = 0))})
    wandb.log({"Forecast Visualization (Last Day)":  wandb.Image(plot_forecasts(ground_truth, forecasts, mask, day = -1))})

    # Extracting hypoxia regions
    ground_truth = (ground_truth < treshold) * 1.0
    forecasts    = (forecasts < treshold) * 1.0

    # Plotting the hypoxia regions
    wandb.log({"Hypoxia Visualization (First Day)": wandb.Image(plot_hypoxia(ground_truth, forecasts, mask, day = 0))})
    wandb.log({"Hypoxia Visualization (Last Day)": wandb.Image(plot_hypoxia(ground_truth, forecasts, mask, day = -1))})

    # Plotting the probability maps
    wandb.log({"Probability Visualization (First Day)": wandb.Image(plot_probability_maps(ground_truth, forecasts, mask, day = 0))})
    wandb.log({"Probability Visualization (Last Day)": wandb.Image(plot_probability_maps(ground_truth, forecasts, mask, day = -1))})

    # Computing global metrics
    compute_precision_recall(ground_truth, forecasts, mask)