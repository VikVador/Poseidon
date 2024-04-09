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
# Pytorch
import torch
import torch.nn as nn


class FCNN(nn.Sequential):
    r"""A fully convolutional neural network"""

    def __init__(self, inputs: int, kernel_size : int = 3, scaling : int = 1):
        super(FCNN, self).__init__()

        # Initialization (Mean and STD)
        self.n_in    = inputs
        self.n_out   = 2
        self.padding = kernel_size // 2

        # ------ Architecture ------
        #
        # Main Layers
        self.conv_init           = nn.Conv2d(self.n_in    , 256 * scaling, kernel_size, padding = self.padding)
        self.conv_intermediate_1 = nn.Conv2d(256 * scaling, 128 * scaling, kernel_size, padding = self.padding)
        self.conv_intermediate_2 = nn.Conv2d(128 * scaling,  64 * scaling, kernel_size, padding = self.padding)
        self.conv_intermediate_3 = nn.Conv2d( 64 * scaling,  32 * scaling, kernel_size, padding = self.padding)
        self.conv_final          = nn.Conv2d( 32 * scaling,    self.n_out, kernel_size, padding = self.padding)

        # Activation function
        self.activation = nn.GELU()

        # Normalization
        self.normalization_init           = nn.BatchNorm2d(self.conv_init.out_channels)
        self.normalization_intermediate_1 = nn.BatchNorm2d(self.conv_intermediate_1.out_channels)
        self.normalization_intermediate_2 = nn.BatchNorm2d(self.conv_intermediate_2.out_channels)
        self.normalization_intermediate_3 = nn.BatchNorm2d(self.conv_intermediate_3.out_channels)

    def forward(self, x):
        x = self.normalization_init(self.activation(self.conv_init(x)))
        x = self.normalization_intermediate_1(self.activation(self.conv_intermediate_1(x)))
        x = self.normalization_intermediate_2(self.activation(self.conv_intermediate_2(x)))
        x = self.normalization_intermediate_3(self.activation(self.conv_intermediate_3(x)))
        x = self.conv_final(x)

        # Retrieiving dimensions (Ease of comprehension)
        b, c, x_res, y_res = x.shape

        # Reshaping the output, i.e. (samples, days, values, x, y)
        return x.reshape(b, self.n_out // 2, 2, x_res, y_res)

    def count_parameters(self,):
        r"""Determines the number of trainable parameters in the model"""
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))
