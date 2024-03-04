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

# Custom libraries
from neural_networks.ENCODER import ENCODER


class FCNN(nn.Sequential):
    r"""A fully convolutional neural network"""

    def __init__(self, problem: str, inputs: int, outputs: int, window_transformation: int = 1, kernel_size : int = 3, scaling : int = 1):
        super(FCNN, self).__init__()

        # Initialization
        self.n_in    = inputs
        self.problem = problem
        self.padding = kernel_size // 2

        # Number of output channels, i.e. times 2 because either mean and std for regression or both classes for classification
        self.n_out   = outputs * 2

        # ------ Architecture ------
        #
        # Temporal Encoder
        self.block_encoder = ENCODER(window_transformation)

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

    def forward(self, x, t):

        # Retrieiving dimensions (Ease of comprehension)
        samples, days, values, variables, x_res, y_res = x.shape

        # ----- Encoding Time -----
        #
        # Applying the encoder
        weights = torch.squeeze(self.block_encoder(t), dim = -1)

        # Applying the weights (except to mesh (dim = 2) and bathymetry (dim = 3))
        for sample in range(samples):
            for value in range(days):
                x[:, value, :, :-3] *= weights[sample, value]

        # Reshaping
        x = x.reshape(samples, days * values * variables, x_res, y_res)

        # ----- Fully Convolutionnal -----
        #
        x = self.normalization_init(self.activation(self.conv_init(x)))
        x = self.normalization_intermediate_1(self.activation(self.conv_intermediate_1(x)))
        x = self.normalization_intermediate_2(self.activation(self.conv_intermediate_2(x)))
        x = self.normalization_intermediate_3(self.activation(self.conv_intermediate_3(x)))
        x = self.conv_final(x)

        # ----- Reshaping -----
        #
        # Reshaping the output, i.e. (b, c, x, y) -> (b, c/2, 2, x, y) for classification
        if self.problem == "classification":

            # Retrieiving dimensions (Ease of comprehension)
            b, c, x_res, y_res = x.shape

            # Reshaping
            x = x.reshape(b, self.n_out // 2, 2, x_res, y_res)

            # Note: The BCELosswithdigits applies a sigmoid function
            #       on the output thus we do not need to apply it ourselves.

        return x

    def count_parameters(self,):
        r"""Determines the number of trainable parameters in the model"""
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))
