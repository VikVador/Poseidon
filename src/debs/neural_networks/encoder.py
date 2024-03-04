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
# A neural network definition to be used as temporal encoder
#
# Pytorch
import torch.nn as nn


class ENCODER(nn.Sequential):
    r"""A neural network used to encode the temporal information of the data and return weights for the input data"""

    def __init__(self, input_size : int):
        super(ENCODER, self).__init__()

        # Defining the layers
        self.linear_in       = nn.Linear(input_size, 256)
        self.linear_middle_1 = nn.Linear(256,        256)
        self.linear_middle_2 = nn.Linear(256,        128)
        self.linear_middle_3 = nn.Linear(128,         64)
        self.linear_middle_4 = nn.Linear(64,          32)
        self.linear_out      = nn.Linear(32,           1)

        # Defining the activation functions
        self.activation = nn.GELU()

        # Defining the softmax function, i.e. (t, values, day) to (t, values, 1) then (t, weights, 1)
        self.softmax = nn.Softmax(dim = 0)

    def forward(self, x):

        # Applying the layers
        x = self.activation(self.linear_in(x))
        x = self.activation(self.linear_middle_1(x))
        x = self.activation(self.linear_middle_2(x))
        x = self.activation(self.linear_middle_3(x))
        x = self.activation(self.linear_middle_4(x))
        x = self.linear_out(x)

        # Applying the softmax function
        return self.softmax(x)

    def count_parameters(self,):
        r"""Determines the number of trainable parameters in the model"""
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))
