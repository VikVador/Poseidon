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

    def __init__(self, inputs: int, outputs: int, problem : str,  kernel_size: int = 3):

        # Initialization
        self.n_in    = inputs
        self.padding = kernel_size // 2
        self.problem = problem

        # Number of output channels,
        self.n_out   = outputs * 2 if problem == "classification" else outputs

        # ------ Architecture ------
        block1 = self._make_subblock(nn.Conv2d(self.n_in, 256, kernel_size, padding = self.padding))
        block2 = self._make_subblock(nn.Conv2d(256,       128, kernel_size, padding = self.padding))
        block3 = self._make_subblock(nn.Conv2d(128,        32, kernel_size, padding = self.padding))
        block4 = self._make_subblock(nn.Conv2d(32,         32, kernel_size, padding = self.padding))
        block5 = self._make_subblock(nn.Conv2d(32,         32, kernel_size, padding = self.padding))
        block6 = self._make_subblock(nn.Conv2d(32,         32, kernel_size, padding = self.padding))
        block7 = self._make_subblock(nn.Conv2d(32,         32, kernel_size, padding = self.padding))
        conv8  =                     nn.Conv2d(32, self.n_out, kernel_size, padding = self.padding)

        # Combining everything together
        super().__init__(*block1, *block2, *block3, *block4, *block5, *block6, *block7, conv8)

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

            return sf(x)

        return x

    def count_parameters(self,):
        print("Model parameters  =", sum(p.numel() for p in self.parameters() if p.requires_grad))