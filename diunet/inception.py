import torch
from torch import nn
import torch.nn.functional as F


class InceptionResBlock(nn.Module):
    """
    A modified residual inception module proposed by the paper to be
    used in the analysis and synthesis path of the U-Net architecture.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        is_output_block=False,
        skip_feature_size=0,
    ):
        """
        Parameters:
        - in_channels: number of channels for input data
        - out_channels: desired number of channels for this block to return
        """
        super(InceptionResBlock, self).__init__()
        self.out_channels = out_channels
        self.is_output_block = is_output_block
        self.skip_feature_size = skip_feature_size

        # layers for 1st branch
        self.branch1_1x1conv = nn.Conv2d(
            in_channels=int(in_channels + skip_feature_size),
            out_channels=in_channels,
            kernel_size=1,
        )
        self.branch1_bn = nn.BatchNorm2d(in_channels)

        # layers for 2nd branch
        self.branch2_1x1conv = nn.Conv2d(
            in_channels=int(in_channels + skip_feature_size),
            out_channels=in_channels,
            kernel_size=1,
        )
        self.branch2_3x3conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding="same",
        )
        self.branch2_bn = nn.BatchNorm2d(in_channels)

        # layers for 3rd branch
        self.branch3_1x1conv = nn.Conv2d(
            in_channels=int(in_channels + skip_feature_size),
            out_channels=in_channels,
            kernel_size=1,
        )
        self.branch3_3x3conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding="same",
        )
        self.branch3_bn1 = nn.BatchNorm2d(in_channels)
        self.branch3_3x3conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding="same",
        )
        self.branch3_bn2 = nn.BatchNorm2d(in_channels)

        self.bottleneck_1x1conv = nn.Conv2d(
            in_channels=in_channels * 3, out_channels=out_channels, kernel_size=1
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x, skip_features=None):
        x_start = x
        if self.skip_feature_size:
            x_start = torch.cat([x_start, skip_features], dim=1)

        x1 = self.branch1_1x1conv(x_start)
        x1 = F.relu(self.branch1_bn(x1))

        x2 = self.branch2_1x1conv(x_start)
        x2 = self.branch2_3x3conv(x2)
        x2 = F.relu(self.branch2_bn(x2))

        x3 = self.branch3_1x1conv(x_start)
        x3 = self.branch3_3x3conv1(x3)
        x3 = F.relu(self.branch3_bn1(x3))
        x3 = self.branch3_3x3conv2(x3)
        x3 = F.relu(self.branch3_bn2(x3))

        x_concat = torch.cat([x1, x2, x3], dim=1)
        x_bottleneck = self.bottleneck_1x1conv(x_concat)
        x_identity = self.downsample(x)

        if self.is_output_block:
            return F.sigmoid(x_bottleneck + x_identity)

        return F.relu(x_bottleneck + x_identity)


class WideInceptionResBlock(nn.Module):
    """
    Another modified residual inception module, but this module is proposed to be
    used in a dense connection block.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(WideInceptionResBlock, self).__init__()

        # 1st branch layers
        self.branch1_1x1conv = nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels, kernel_size=1
        )
        self.branch1_bn = nn.BatchNorm2d(in_channels)

        # 2nd branch layers
        self.branch2_mp = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch2_1x1conv = nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels, kernel_size=1
        )
        self.branch2_bn = nn.BatchNorm2d(in_channels)

        # 3rd branch layers
        self.branch3_1x1conv = nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels, kernel_size=1
        )
        self.branch3_3x1conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(3, 1),
            padding="same",
        )
        self.branch3_bn1 = nn.BatchNorm2d(in_channels)
        self.branch3_1x3conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(1, 3),
            padding="same",
        )
        self.branch3_bn2 = nn.BatchNorm2d(in_channels)

        self.bottleneck_1x1conv = nn.Conv2d(
            in_channels=in_channels * 3, out_channels=out_channels, kernel_size=1
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x1 = self.branch1_1x1conv(x)
        x1 = F.relu(self.branch1_bn(x1))

        x2 = self.branch2_mp(x)
        x2 = self.branch2_1x1conv(x2)
        x2 = F.relu(self.branch2_bn(x2))

        x3 = self.branch3_1x1conv(x)
        x3 = self.branch3_3x1conv(x3)
        x3 = F.relu(self.branch3_bn1(x3))
        x3 = self.branch3_1x3conv(x3)
        x3 = F.relu(self.branch3_bn2(x3))

        x_concat = torch.cat([x1, x2, x3], dim=1)
        x_bottleneck = self.bottleneck_1x1conv(x_concat)
        x_identity = self.downsample(x)

        return F.relu(x_bottleneck + x_identity)
