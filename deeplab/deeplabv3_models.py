from torch import nn
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
from diunet.inception import InceptionResBlock
from diunet.dense_inception import DenseInceptionBlock


def deeplabv3_modded(backbone="resnet50"):
    if backbone == "resnet50":
        model = deeplabv3_resnet50(num_classes=1)
    elif backbone == "resnet101":
        model = deeplabv3_resnet101(num_classes=1)
    else:
        raise Exception("Invalid backbone")

    model.backbone.conv1 = nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    model.classifier.add_module("output", nn.Sigmoid())

    return model


def inception_deeplabv3(backbone="resnet50", inception_module_count=1):
    model = deeplabv3_modded(backbone)

    # remove original classification head and add custom inception modules
    del model.classifier[1:3]

    inception_modules = nn.Sequential(
        *[
            InceptionResBlock(in_channels=256, out_channels=256)
            for _ in range(inception_module_count)
        ]
    )
    model.classifier[1] = inception_modules

    return model
