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

class LayerNormalization(nn.Module):
    r"""Custom Layer Normalization module"""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor):
        var, mean = torch.var_mean(x, dim = self.dim, keepdim = True)
        return (x - mean)/torch.sqrt(var + self.eps)

class TimeResidual_Block(nn.Module):
    r"""A time residual block for UNET"""

    def __init__(self, input_channels: int, frequencies: int):
        super(TimeResidual_Block, self).__init__()

        # Initializations
        self.frequencies   = frequencies
        self.activation    = nn.SiLU()
        self.normalization = LayerNormalization(dim = 1)
        self.variance      = torch.sqrt(torch.tensor(2))

        # Temporal Projection
        self.time_mixing     = nn.Linear(in_features = self.frequencies * 2 * 3, out_features = self.frequencies * 2)
        self.time_projection = nn.Linear(in_features = self.frequencies * 2,     out_features = input_channels, bias = False)

        # Convolutions
        self.conv1 = nn.Conv2d(in_channels  = input_channels,
                               out_channels = input_channels,
                               kernel_size  = 3,
                               stride       = 1,
                               padding      = 1)

        self.conv2 = nn.Conv2d(in_channels  = input_channels,
                               out_channels = input_channels,
                               kernel_size  = 3,
                               stride       = 1,
                               padding      = 1)

    def forward(self, x, time):

        # -------------------
        #        Time
        # -------------------
        # 1. Initial information
        b, c, x_res, y_res = x.shape

        # 2. Temporal Mixing and activation
        encoded_time = self.time_mixing(time)
        encoded_time = self.activation(encoded_time)

        # 3. Temporal Projection
        encoded_time = self.time_projection(encoded_time)

        # 4. Reshaping the time encoding
        encoded_time = encoded_time[:, :, None, None]

        # 5. Creating the grids
        encoded_time = encoded_time.expand(-1, -1, x_res, y_res)

        # -------------------
        #        Spatial
        # -------------------
        # 1. Adding temporal information (broadcasting)
        x_residual = x + encoded_time

        # 2. Normalization
        x_residual = self.normalization(x_residual)

        # 3. Convolution (1)
        x_residual = self.conv1(x_residual)

        # 4. Activation
        x_residual = self.activation(x_residual)

        # 5. Convolution (2)
        x_residual = self.conv2(x_residual)

        # 6. Adding the residual
        x = x + x_residual

        # 7. Keeping unit variance
        return x / self.variance

    def count_parameters(self,):
        r"""Determines the number of trainable parameters in the model"""
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

class TimeResidual_UNET(nn.Module):
    r"""A time residual UNET for time series forecasting"""

    def __init__(self, input_channels: int, output_channels: int, frequencies: int, number_gaussians: int, scaling: int = 1):
        super(TimeResidual_UNET, self).__init__()

        # Initializations
        self.frequencies      = frequencies
        self.input_channels   = input_channels
        self.output_channels  = output_channels
        self.number_gaussians = number_gaussians

        # 1. Input (lifting)
        self.input_conv = nn.Conv2d(in_channels = self.input_channels, out_channels = 32 * scaling, kernel_size = 3, stride = 1, padding = 1)

        # 2. Downsampling
        #
        # Time Residual Blocks (1)
        self.downsample_11_residuals = TimeResidual_Block(input_channels  = 32 * scaling,     frequencies = self.frequencies)
        self.downsample_12_residuals = TimeResidual_Block(input_channels  = 32 * scaling,     frequencies = self.frequencies)
        self.downsample_21_residuals = TimeResidual_Block(input_channels  = 32 * scaling * 2, frequencies = self.frequencies)
        self.downsample_22_residuals = TimeResidual_Block(input_channels  = 32 * scaling * 2, frequencies = self.frequencies)
        self.downsample_31_residuals = TimeResidual_Block(input_channels  = 32 * scaling * 4, frequencies = self.frequencies)
        self.downsample_32_residuals = TimeResidual_Block(input_channels  = 32 * scaling * 4, frequencies = self.frequencies)
        self.downsample_41_residuals = TimeResidual_Block(input_channels  = 32 * scaling * 8, frequencies = self.frequencies)
        self.downsample_42_residuals = TimeResidual_Block(input_channels  = 32 * scaling * 8, frequencies = self.frequencies)

        # Convolutions (downsampling)
        self.downsample_1_conv = nn.Conv2d(in_channels = 32 * scaling,     out_channels = 32 * scaling * 2, kernel_size = 2, stride = 2)
        self.downsample_2_conv = nn.Conv2d(in_channels = 32 * scaling * 2, out_channels = 32 * scaling * 4, kernel_size = 2, stride = 2)
        self.downsample_3_conv = nn.Conv2d(in_channels = 32 * scaling * 4, out_channels = 32 * scaling * 8, kernel_size = 2, stride = 2)

        # 3. Upsampling
        #
        # Used for upsampling instead of transposed convolutions
        self.upsample = nn.Upsample(scale_factor = (2, 2))

        # Convolutions (projection)
        self.projection_1 = nn.Conv2d(in_channels = 32 * scaling * (8 + 4), out_channels = 32 * scaling * 4, kernel_size = 3, padding = 1)
        self.projection_2 = nn.Conv2d(in_channels = 32 * scaling * (4 + 2), out_channels = 32 * scaling * 2, kernel_size = 3, padding = 1)
        self.projection_3 = nn.Conv2d(in_channels = 32 * scaling * (2 + 1), out_channels = 32 * scaling    , kernel_size = 3, padding = 1)

        # Time Residual Blocks (2)
        self.upsample_11_residuals = TimeResidual_Block(input_channels = 32 * scaling * 4, frequencies = self.frequencies)
        self.upsample_12_residuals = TimeResidual_Block(input_channels = 32 * scaling * 4, frequencies = self.frequencies)
        self.upsample_21_residuals = TimeResidual_Block(input_channels = 32 * scaling * 2, frequencies = self.frequencies)
        self.upsample_22_residuals = TimeResidual_Block(input_channels = 32 * scaling * 2, frequencies = self.frequencies)
        self.upsample_31_residuals = TimeResidual_Block(input_channels = 32 * scaling    , frequencies = self.frequencies)
        self.upsample_32_residuals = TimeResidual_Block(input_channels = 32 * scaling    , frequencies = self.frequencies)

        # 4. Output (We use a linear to mix accross channels, a convolution mix spatially and introduce bias at the corners)
        self.output_linear = nn.Linear(in_features = 32 * scaling, out_features = self.output_channels, bias = False)

        # Normalization
        self.normalization = LayerNormalization(dim = 1)


    def forward(self, x, time):

        # 1. Lifting
        x = self.input_conv(x)

        # 2. Downsampling
        x = self.downsample_11_residuals(x, time)
        x = self.downsample_12_residuals(x, time)

        x1 = self.downsample_1_conv(x)
        x1 = self.downsample_21_residuals(x1, time)
        x1 = self.downsample_22_residuals(x1, time)

        x2 = self.downsample_2_conv(x1)
        x2 = self.downsample_31_residuals(x2, time)
        x2 = self.downsample_32_residuals(x2, time)

        x3 = self.downsample_3_conv(x2)
        x3 = self.downsample_41_residuals(x3, time)
        x3 = self.downsample_42_residuals(x3, time)
        x3 = self.normalization(x3)

        # 3. Upsampling
        x3 = self.normalization(x3) # Good Practice for conditioning the data with diffusion, etc.
        x3 = self.upsample(x3)

        x2 = torch.cat([x3, x2], dim = 1)
        x2 = self.projection_1(x2)
        x2 = self.upsample_11_residuals(x2, time)
        x2 = self.upsample_12_residuals(x2, time)
        x2 = self.normalization(x2)
        x2 = self.upsample(x2)

        x1 = torch.cat([x2, x1], dim = 1)
        x1 = self.projection_2(x1)
        x1 = self.upsample_21_residuals(x1, time)
        x1 = self.upsample_22_residuals(x1, time)
        x1 = self.normalization(x1)
        x1 = self.upsample(x1)

        x = torch.cat([x1, x], dim = 1)
        x = self.projection_3(x)
        x = self.upsample_31_residuals(x, time)
        x = self.upsample_32_residuals(x, time)

        # 4. Output
        x = self.output_linear(torch.permute(x, (0, 2, 3, 1)))

        # 5. Adding separate channels for mean and log(var)
        x = torch.permute(x, (0, 3, 1, 2))
        b, _, x_res, y_res = x.shape

        # Reshaping the output, i.e. (samples, days, values (mean|log(var)|pi, x, y)
        return x.reshape(b, self.output_channels // (3 * self.number_gaussians), self.number_gaussians, 3, x_res, y_res)

    def count_parameters(self,):
        r"""Determines the number of trainable parameters in the model"""
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))