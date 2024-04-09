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
# Function used to create custom losses
#
# Pytorch
import torch


def loss_regression(y_pred : torch.Tensor, y_true : torch.Tensor, mask : torch.Tensor):
    r"""Regression loss (Mean and Standard Deviation)"""

    # Extracting the mean and the log variance)
    ground_truth = y_true[:, 0, 0, mask == 1]
    mean         = y_pred[:, 0, 0, mask == 1]
    log_variance = y_pred[:, 0, 1, mask == 1]
    variance     = torch.exp(log_variance)

    # Computing the negative log likelihood (element wise)
    error_pixel_wise = -log_variance - ( ((ground_truth - mean) ** 2) / (2 * variance) )

    # Computing the negative log likelihood (all) and averaging over all pixels to get the final loss
    error = - torch.nansum(error_pixel_wise) / torch.numel(error_pixel_wise)

    # Summing and averaging over the observed region
    return error
