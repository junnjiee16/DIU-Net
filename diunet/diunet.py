from torch import nn
from diunet.inception import InceptionResBlock
from diunet.dense_inception import DenseInceptionBlock
from diunet.sampling_blocks import DownsamplingBlock, UpsamplingBlock


class DIUNet(nn.Module):
    """
    PyTorch implementation of the Dense Inception U-Net (DIU-Net)
    """

    def __init__(self, scale=1):
        super(DIUNet, self).__init__()

        # analysis path
        self.analysis_inception1 = InceptionResBlock(in_channels=1, out_channels=64)
        self.analysis_downsampling1 = DownsamplingBlock(in_channels=64)
        self.analysis_inception2 = InceptionResBlock(in_channels=64, out_channels=128)
        self.analysis_downsampling2 = DownsamplingBlock(in_channels=128)
        self.analysis_inception3 = InceptionResBlock(in_channels=128, out_channels=256)
        self.analysis_downsampling3 = DownsamplingBlock(in_channels=256)

        self.analysis_denseinception = DenseInceptionBlock(
            in_channels=256, out_channels=512, depth=12
        )
        self.analysis_downsampling4 = DownsamplingBlock(in_channels=512)

        # middle block
        self.middle_denseinception = DenseInceptionBlock(
            in_channels=512, out_channels=1024, depth=24
        )

        # synthesis path
        self.synthesis_upsampling1 = UpsamplingBlock(in_channels=1024)
        self.synthesis_denseinception = DenseInceptionBlock(
            in_channels=1024, out_channels=512, depth=12
        )

        self.synthesis_upsampling2 = UpsamplingBlock(in_channels=512)
        self.synthesis_inception2 = InceptionResBlock(in_channels=512, out_channels=256)
        self.synthesis_upsampling3 = UpsamplingBlock(in_channels=256)
        self.synthesis_inception3 = InceptionResBlock(in_channels=256, out_channels=128)
        self.synthesis_upsampling4 = UpsamplingBlock(in_channels=128)
        self.synthesis_inception4 = InceptionResBlock(in_channels=128, out_channels=64)

        self.synthesis_inception5 = InceptionResBlock(in_channels=64, out_channels=32)
        self.synthesis_inception6 = InceptionResBlock(in_channels=32, out_channels=2)

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
        x = self.synthesis_inception2(x)
        x = self.synthesis_upsampling3(x)
        x = self.synthesis_inception3(x)
        x = self.synthesis_upsampling4(x)
        x = self.synthesis_inception4(x)
        x = self.synthesis_inception5(x)
        x = self.synthesis_inception6(x)

        return x
