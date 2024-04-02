import torch
from torch import nn
from diunet.inception import WideInceptionResBlock


class DenseInceptionBlock(nn.Module):
    """
    Proposes adding Inception modules into a dense connection block. Adds residual
    connections at every bottleneck layer.
    """

    def __init__(self, in_channels: int, out_channels: int, depth: int):
        super(DenseInceptionBlock, self).__init__()

        self.depth = depth
        self.inception_blocks = nn.ModuleList(
            [WideInceptionResBlock(in_channels, in_channels) for _ in range(depth)]
        )

        self.bottleneck_layers = nn.ModuleList(
            [
                (
                    nn.Conv2d(
                        in_channels=in_channels * (i + 1),
                        out_channels=out_channels,
                        kernel_size=1,
                    )
                    if i + 1 == depth
                    else nn.Conv2d(
                        in_channels=in_channels * (i + 1),
                        out_channels=in_channels,
                        kernel_size=1,
                    )
                )
                for i in range(depth)
            ]
        )

    def forward(self, x):
        feature_maps = []

        for i in range(self.depth):
            inceptionModule = self.inception_blocks[i]
            bottleneckLayer = self.bottleneck_layers[i]

            feature_maps.append(inceptionModule(x))

            if len(feature_maps) > 1:
                x_concat = torch.cat(feature_maps, dim=1)
            else:
                x_concat = feature_maps[0]

            x = bottleneckLayer(x_concat)

        return x
