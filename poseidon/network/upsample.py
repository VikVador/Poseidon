import torch.nn as nn

from torch import Tensor

# isort: split
from poseidon.network.tools import reshape, unshape


class UpsampleBlock(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()

        self.block = nn.Upsample(scale_factor=scale_factor, mode="nearest")

    def forward(self, x: Tensor) -> Tensor:
        x, _, original_shape = reshape("spatial", x)
        x = self.block(x)
        x = unshape("spatial", x, original_shape)
        return x
