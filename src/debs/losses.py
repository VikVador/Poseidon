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
# Definition of the loss(es) used during training
#
import torch
import torch.nn as nn
import numpy as np


def forecasting_loss(y_true: torch.Tensor, y_pred: torch.Tensor, mask: np.array):
    """Used to compute the forecasting loss for a Gaussian Mixture Model"""

    def mixture_density_network_loss(y_true: torch.Tensor, y_pred: torch.Tensor):
        """Used to compute the loss on an individual day for a MDN"""

        # Extracting the values
        mean, log_var, pi = y_pred[:, :, 0], y_pred[:, :, 1], y_pred[:, :, 2]

        # Normalizing the constants and computing the log
        log_pi = nn.functional.log_softmax(pi, dim = 1)

        # Adding missing dimension for broadcasting
        y_true = y_true[:, None, :].expand(-1, mean.shape[1], -1)

        # Computing the loss
        loss = torch.logsumexp(log_pi -  log_var - ((y_true - mean) ** 2)/torch.exp(log_var), dim = 1)

        # Averaging loss over all pixel space
        return - torch.nanmean(loss)

    # Extracting the sea values
    y_true = y_true[:, :,       mask[0] == 1]
    y_pred = y_pred[:, :, :, :, mask[0] == 1]

    # Extracting dimensions for ease of comprehension
    batch, number_days, number_gaussians, values, _ = y_pred.shape

    # Computing the loss for each individual day
    loss_per_day = [mixture_density_network_loss(y_true[:, i], y_pred[:, i]) for i in range(number_days)]

    # Returning the total loss and loss per day
    return torch.sum(torch.hstack(loss_per_day)), loss_per_day
