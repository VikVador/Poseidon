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
import torch.nn as nn


class FCNN(nn.Sequential):
    r"""A fully convolutional neural network"""

    def __init__(self, problem : str, inputs : int, outputs : int,  kernel_size : int = 3, scaling : int = 1):

        # Initialization
        self.n_in    = inputs
        self.padding = kernel_size // 2
        self.problem = problem

        # Number of output channels,
        self.n_out   = outputs * 2 if problem == "classification" else outputs

        # ------ Architecture ------
        #
        # Main Layers
        block_init           = self._make_subblock(nn.Conv2d(self.n_in,        256, kernel_size, padding = self.padding))
        block_intermediate_1 = self._make_subblock(nn.Conv2d(      256,        128, kernel_size, padding = self.padding))
        block_intermediate_2 = self._make_subblock(nn.Conv2d(      128,         64, kernel_size, padding = self.padding))
        block_intermediate_3 = self._make_subblock(nn.Conv2d(       64,         32, kernel_size, padding = self.padding))
        block_final          =                    [nn.Conv2d(       32, self.n_out, kernel_size, padding = self.padding)]

        # Stores additionnal layers to increase capacity
        block_level_1, block_level_2, block_level_3, block_level_4 = list(), list(), list(), list()

        # Adding layers
        for i in range(scaling - 1):
            block_level_1 += self._make_subblock(nn.Conv2d(256, 256, kernel_size, padding = self.padding))
            block_level_2 += self._make_subblock(nn.Conv2d(128, 128, kernel_size, padding = self.padding))
            block_level_3 += self._make_subblock(nn.Conv2d( 64,  64, kernel_size, padding = self.padding))
            block_level_4 += self._make_subblock(nn.Conv2d( 32,  32, kernel_size, padding = self.padding))

        # Combining everything together
        super().__init__(*block_init, *block_level_1, *block_intermediate_1,
                                      *block_level_2, *block_intermediate_2,
                                      *block_level_3, *block_intermediate_3,
                                      *block_level_4,          *block_final)

    def _make_subblock(self, conv):
        return [conv, nn.GELU(), nn.BatchNorm2d(conv.out_channels)]

    def forward(self, x):

        # Forward pass
        x = super().forward(x)

        # Reshaping the output, i.e. (b, c, x, y) -> (b, c/2, 2, x, y) for classification
        if self.problem == "classification":

            # Retrieiving dimensions (Ease of comprehension)
            b, c, x_res, y_res = x.shape

            # Reshaping
            x = x.reshape(b, self.n_out // 2, 2, x_res, y_res)

            # Turning to probabilities
            sf = nn.Softmax(dim = 2)
            x  = sf(x)

        return x

    def count_parameters(self,):
        r"""Determines the number of trainable parameters in the model"""
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))