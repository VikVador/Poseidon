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


class UNET(nn.Module):
    r"""A simple UNET architecture"""

    def __init__(self, problem : str, inputs : int, outputs : int, window_transformation: int = 1, kernel_size : int = 3, scaling : int = 1):
        super(UNET, self).__init__()

        # Initialization
        self.n_in    = inputs
        self.padding = kernel_size // 2
        self.kernel  = kernel_size
        self.problem = problem
        features     = 8 + 4 * (scaling - 1)

        # Number of output channels,
        self.n_out = outputs * 2

        # ------ Architecture ------
        #
        # Temporal Encoder
        self.block_encoder = ENCODER(window_transformation)

        # Main Layers
        self.pool1      = nn.MaxPool2d(                                   kernel_size = 2, stride = 2)
        self.pool2      = nn.MaxPool2d(                                   kernel_size = 2, stride = 2)
        self.pool3      = nn.MaxPool2d(                                   kernel_size = 2, stride = 2)
        self.pool4      = nn.MaxPool2d(                                   kernel_size = 2, stride = 2)
        self.upconv4    = nn.ConvTranspose2d(features * 16, features * 8, kernel_size = 2, stride = 2)
        self.upconv3    = nn.ConvTranspose2d(features * 8,  features * 4, kernel_size = 2, stride = 2)
        self.upconv2    = nn.ConvTranspose2d(features * 4,  features * 2, kernel_size = 2, stride = 2)
        self.upconv1    = nn.ConvTranspose2d(features * 2,  features,     kernel_size = 2, stride = 2)

        # Convolutional blocks
        self.encoder1   = self._make_subblock(self.n_in,          features)
        self.encoder2   = self._make_subblock(features,       features * 2)
        self.encoder3   = self._make_subblock(features  * 2,  features * 4)
        self.encoder4   = self._make_subblock(features  * 4,  features * 8)
        self.bottleneck = self._make_subblock(features  * 8, features * 16)
        self.decoder4   = self._make_subblock(features * 16,  features * 8)
        self.decoder3   = self._make_subblock(features *  8,  features * 4)
        self.decoder2   = self._make_subblock(features *  4,  features * 2)
        self.decoder1   = self._make_subblock(features *  2,      features)

        # Final convolutional layer
        self.conv       = nn.Conv2d(in_channels = features, out_channels = self.n_out, kernel_size = 1)

    def _make_subblock(self, inputs : int, features : int):
        r"""Returns a convolutional block"""
        return nn.Sequential(
            nn.Conv2d(in_channels  = inputs,
                      out_channels = features,
                      kernel_size  = self.kernel,
                      padding      = self.padding,
                      bias         = False),
            nn.BatchNorm2d(num_features = features),
            nn.GELU(),
            nn.Conv2d(in_channels  = features,
                      out_channels = features,
                      kernel_size  = self.kernel,
                      padding      = self.padding,
                      bias         = False),
            nn.BatchNorm2d(num_features = features),
            nn.GELU())

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

        # ----- U-NET -----
        #
        enc1       = self.encoder1(x)
        enc2       = self.encoder2(self.pool1(enc1))
        enc3       = self.encoder3(self.pool2(enc2))
        enc4       = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))
        dec4       = torch.cat((self.upconv4(bottleneck), enc4), dim = 1)
        dec4       = self.decoder4(dec4)
        dec3       = torch.cat((self.upconv3(dec4), enc3), dim = 1)
        dec3       = self.decoder3(dec3)
        dec2       = torch.cat((self.upconv2(dec3), enc2), dim = 1)
        dec2       = self.decoder2(dec2)
        dec1       = torch.cat((self.upconv1(dec2), enc1), dim = 1)
        dec1       = self.decoder1(dec1)
        x          = self.conv(dec1)

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
