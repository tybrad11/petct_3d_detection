
from model_parts import *
from torch import nn


class UNet3D(nn.Module):
    """
    Base class for standard and residual UNet.

    Args:
        input_channels=2,
        layer_channels = [32, 64, 128, 256],   #channels at each layer
        n_classes=2,
        trilinear=False
    """

    def __init__(self, input_channels=2,
                 layer_channels = [32, 64, 128, 256, 320],   #channels at each layer
                 n_classes=2,
                 norm_method = 'instance',      #'instance' or 'batchnorm
                 trilinear=False):
        super(UNet3D, self).__init__()
        self.n_channels = input_channels
        self.n_classes = n_classes
        self.trilinear = trilinear

        self.inc = DoubleConv(input_channels, layer_channels[0], norm_method)
        self.down1 = Down(layer_channels[0], layer_channels[1], norm_method)
        self.down2 = Down(layer_channels[1], layer_channels[2], norm_method)
        self.down3 = Down(layer_channels[2], layer_channels[3], norm_method)
        factor = 2 if trilinear else 1
        self.down4 = Down(layer_channels[3], (layer_channels[4]), norm_method)
        self.up1 = Up((layer_channels[4]+layer_channels[3]), layer_channels[3], norm_method, trilinear)
        self.up2 = Up(layer_channels[3]+layer_channels[2], layer_channels[2], norm_method, trilinear)
        self.up3 = Up(layer_channels[2]+layer_channels[1], layer_channels[1], norm_method, trilinear)
        self.up4 = Up(layer_channels[1]+layer_channels[0], layer_channels[0], norm_method, trilinear)
        self.outc = OutConv(layer_channels[0], n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

