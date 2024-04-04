from torch import nn
from diunet.inception import InceptionResBlock
from diunet.dense_inception import DenseInceptionBlock
from diunet.sampling_blocks import DownsamplingBlock, UpsamplingBlock


class DIUNet(nn.Module):
    """
    PyTorch implementation of the Dense Inception U-Net (DIU-Net)
    """

    def __init__(self, out_channels: int, channel_scale=1, dense_block_depth_scale=1):
        super(DIUNet, self).__init__()

        # analysis path
        self.analysis_inception1 = InceptionResBlock(
            in_channels=1, out_channels=int(64 * channel_scale)
        )
        self.analysis_downsampling1 = DownsamplingBlock(in_channels=int(64 * channel_scale))
        self.analysis_inception2 = InceptionResBlock(
            in_channels=int(64 * channel_scale), out_channels=int(128 * channel_scale)
        )
        self.analysis_downsampling2 = DownsamplingBlock(in_channels=int(128 * channel_scale))
        self.analysis_inception3 = InceptionResBlock(
            in_channels=int(128 * channel_scale), out_channels=int(256 * channel_scale)
        )
        self.analysis_downsampling3 = DownsamplingBlock(in_channels=int(256 * channel_scale))

        # analysis path dense inception block
        self.analysis_denseinception = DenseInceptionBlock(
            in_channels=int(256 * channel_scale),
            out_channels=int(512 * channel_scale),
            depth=int(12 * dense_block_depth_scale),
        )
        self.analysis_downsampling4 = DownsamplingBlock(in_channels=int(512 * channel_scale))

        # middle dense block
        self.middle_denseinception = DenseInceptionBlock(
            in_channels=int(512 * channel_scale),
            out_channels=int(1024 * channel_scale),
            depth=int(24 * dense_block_depth_scale),
        )

        # synthesis path dense inception block
        self.synthesis_upsampling1 = UpsamplingBlock(in_channels=int(1024 * channel_scale))
        self.synthesis_denseinception = DenseInceptionBlock(
            in_channels=int(1024 * channel_scale),
            out_channels=int(512 * channel_scale),
            depth=int(12 * dense_block_depth_scale),
        )

        # synthesis path
        self.synthesis_upsampling2 = UpsamplingBlock(in_channels=int(512 * channel_scale))
        self.synthesis_inception1 = InceptionResBlock(
            in_channels=int(512 * channel_scale), out_channels=int(256 * channel_scale)
        )
        self.synthesis_upsampling3 = UpsamplingBlock(in_channels=int(256 * channel_scale))
        self.synthesis_inception2 = InceptionResBlock(
            in_channels=int(256 * channel_scale), out_channels=int(128 * channel_scale)
        )
        self.synthesis_upsampling4 = UpsamplingBlock(in_channels=int(128 * channel_scale))
        self.synthesis_inception3 = InceptionResBlock(
            in_channels=int(128 * channel_scale), out_channels=int(64 * channel_scale)
        )

        self.synthesis_inception4 = InceptionResBlock(
            in_channels=int(64 * channel_scale), out_channels=int(32 * channel_scale)
        )
        # final output block
        self.synthesis_inception5 = InceptionResBlock(
            in_channels=int(32 * channel_scale), out_channels=out_channels
        )

    def forward(self, x):
        x = self.analysis_inception1(x)
        x = self.analysis_downsampling1(x)
        x = self.analysis_inception2(x)
        x = self.analysis_downsampling2(x)
        x = self.analysis_inception3(x)
        x = self.analysis_downsampling3(x)
        x = self.analysis_denseinception(x)
        x = self.analysis_downsampling4(x)

        x = self.middle_denseinception(x)

        x = self.synthesis_upsampling1(x)
        x = self.synthesis_denseinception(x)
        x = self.synthesis_upsampling2(x)
        x = self.synthesis_inception1(x)
        x = self.synthesis_upsampling3(x)
        x = self.synthesis_inception2(x)
        x = self.synthesis_upsampling4(x)
        x = self.synthesis_inception3(x)
        x = self.synthesis_inception4(x)
        x = self.synthesis_inception5(x)

        return x
