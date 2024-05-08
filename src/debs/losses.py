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
import numpy as np


def forecasting_loss(y_true: torch.Tensor, y_pred: torch.Tensor, mask: np.array, lambdas: list = None):
    """Used to compute the forecasting loss"""

    # Security
    if lambdas != None:
        assert len(lambdas) == y_true.shape[1], "ERROR (forecasting_loss), The number of multipliers should be equal to the number of forecasted days"

    def loss(y_true: torch.Tensor, y_pred_mean: torch.Tensor, y_pred_log_variance: torch.Tensor):
        """Used to compute the loss on an individual day"""

        # Computing loss for each pixel
        loss_pixelwise = - y_pred_log_variance -  ((y_true - y_pred_mean) ** 2)/torch.exp(y_pred_log_variance)

        # Computing average loss
        return - torch.nansum(loss_pixelwise) / torch.numel(loss_pixelwise)

    # Extracting sea values
    y_true = y_true[:, :,    mask[0] == 1]
    y_pred = y_pred[:, :, :, mask[0] == 1]

    # Extracting the mean and log variance
    y_pred_mean         = y_pred[:, :, 0]
    y_pred_log_variance = y_pred[:, :, 1]

    # Extracting information
    batch_size, number_days, number_values = y_true.shape

    # Computing the loss for each indivual day
    loss_per_day = [loss(y_true[:, i], y_pred_mean[:, i], y_pred_log_variance[:, i]) for i in range(number_days)]

    # Applying the multipliers
    if lambdas != None:
        loss_per_day = [loss_per_day[i] * lambdas[i] for i in range(number_days)]

    # Returning the total loss and loss per day
    return torch.sum(torch.hstack(loss_per_day)), loss_per_day
