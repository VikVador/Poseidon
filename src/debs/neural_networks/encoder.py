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


class encoder(nn.Sequential):
    r"""A neural network used to encode the temporal information of the data and return weights for the input data"""

    def __init__(self, input_size : int, output_size : int):
        super(encoder, self).__init__()

        # Defining the layers
        self.linear_in     = nn.Linear(input_size,     input_size * 2)
        self.linear_middle = nn.Linear(input_size * 2, input_size * 2)
        self.linear_out    = nn.Linear(input_size * 2,    output_size)

        # Defining the activation functions
        self.activation = nn.GELU()

        # Defining the softmax function, i.e. (t, values, day) to (t, values, 1) then (t, weights, 1)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):

        # Applying the layers
        x = self.activation(self.linear_in(x))
        x = self.activation(self.linear_middle(x))
        x = self.linear_out(x)

        # Applying the softmax function
        return self.softmax(x)

    def count_parameters(self,):
        r"""Determines the number of trainable parameters in the model"""
        return int(0)