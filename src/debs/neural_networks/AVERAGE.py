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
from tools import to_device

# Pytorch
import torch
import torch.nn as nn


class AVERAGE(nn.Sequential):
    r"""A 'neural network' that predicts the pixel temporal average (act as a baseline)"""

    def __init__(self, average : torch.Tensor, outputs: int, batch_size : int, device):
        super(AVERAGE, self).__init__()

        # Storing information
        self.outputs = outputs
        self.bs      = batch_size
        self.average = self.process(average)
        self.device  = device

        # Dummy feature
        self.layer = nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return to_device(self.average[:x.shape[0]], self.device)

    def process(self, x : torch.Tensor):
        r"""Used to format the output to the correct shape"""

        # Adding number of forecasted days
        x = torch.unsqueeze(x, dim = 0) if self.outputs == 1 else \
            torch.stack([x for i in range(self.outputs)], dim = 0)

        # Adding batch size
        return torch.stack([x for i in range(self.bs)], dim = 0)

    def count_parameters(self,):
        print("Model parameters  =", sum(p.numel() for p in self.parameters() if p.requires_grad))