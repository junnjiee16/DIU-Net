import torch
from torch import nn
import torch.nn.functional as F


class DownsamplingBlock(nn.Module):
    """
    Down-sampling block for U-net architecture. Halves input feature map's dimensions, assuming
    that width and height are equal.
    """

    def __init__(self, in_channels: int):
        super(DownsamplingBlock, self).__init__()

        self.branch1_mp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.branch2_1x1conv = nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels, kernel_size=1
        )
        self.branch2_3x3conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.branch3_1x1conv = nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels, kernel_size=1
        )
        self.branch3_3x3conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1
        )
        self.branch3_3x3conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.final_1x1conv = nn.Conv2d(
            in_channels=in_channels * 3, out_channels=in_channels, kernel_size=1
        )

    def forward(self, x):
        x1 = self.branch1_mp(x)

        x2 = self.branch2_1x1conv(x)
        x2 = F.relu(self.branch2_3x3conv(x2))

        x3 = self.branch3_1x1conv(x)
        x3 = F.relu(self.branch3_3x3conv1(x3))
        x3 = F.relu(self.branch3_3x3conv2(x3))

        x_cat = torch.cat([x1, x2, x3], dim=1)
        return self.final_1x1conv(x_cat)


class UpsamplingBlock(nn.Module):
    """
    Up-sampling block for U-net architecture. Increases input feature map's dimensions by
    a factor of 2, assuming that width and height are equal.
    """

    def __init__(self, in_channels: int):
        super(UpsamplingBlock, self).__init__()

        self.branch2_1x1convtran = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=in_channels, kernel_size=1
        )
        self.branch2_3x3convtran = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )

        self.branch3_1x1convtran = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=in_channels, kernel_size=1
        )
        self.branch3_3x3convtran1 = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=in_channels, kernel_size=3
        )
        self.branch3_3x3convtran2 = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=2,
            padding=3,
            output_padding=1,
        )

        self.final_1x1convtran = nn.ConvTranspose2d(
            in_channels=in_channels * 3, out_channels=in_channels, kernel_size=1
        )

    def forward(self, x):
        x1 = F.interpolate(x, scale_factor=2, mode="nearest")

        x2 = self.branch2_1x1convtran(x)
        x2 = F.relu(self.branch2_3x3convtran(x2))

        x3 = self.branch3_1x1convtran(x)
        x3 = F.relu(self.branch3_3x3convtran1(x3))
        x3 = F.relu(self.branch3_3x3convtran2(x3))

        x_cat = torch.cat([x1, x2, x3], dim=1)
        return self.final_1x1convtran(x_cat)
