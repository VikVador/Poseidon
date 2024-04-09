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
# A neural network definition to be used as emulator
#
import numpy as np
from tools import to_device

# Pytorch
import torch
import torch.nn as nn


class AVERAGE(nn.Sequential):
    r"""A 'neural network' that predicts the pixel temporal average (baseline)"""

    def __init__(self, data_output : np.array, device : str, kwargs : dict):
        super(AVERAGE, self).__init__()

        # Extracting information
        dataset_size     = kwargs['Datasets Size']

        # Retrieiving dimensions
        t, x, y = data_output.shape

        # Number of training samples
        train_samples = int(t * dataset_size[0])

        # Extracting samples
        training_samples = torch.from_numpy(data_output[: train_samples, :, :])

        # Predicting the average and log of variance
        average_output = torch.mean(training_samples, dim = 0)
        std_output     = torch.log(torch.var(training_samples, dim = 0))

        # Stacking
        average_output = torch.stack([average_output, std_output])

        # Storing information
        self.bs      = kwargs['Batch Size']
        self.average = self.process(average_output)
        self.device  = device

        # Dummy feature (It plays no role whatsoever, it is just a placeholder to make the model work with the trainer)
        self.layer = nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return to_device(self.average[:x.shape[0]], self.device)

    def process(self, x : torch.Tensor):
        r"""Used to format the output to the correct shape"""

        # Adding missing dimensions
        x = torch.unsqueeze(x, dim = 0)

        # Adding batch size
        x = torch.stack([x for i in range(self.bs)], dim = 0)

        return x

    def count_parameters(self,):
        r"""Determines the number of trainable parameters in the model"""
        return int(0)