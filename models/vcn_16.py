import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import UpSample
from .vgg_base import VGG11Encoder


class VCN16(nn.Module):
    """
    Fully Connected Convolutional Neural Net with VGG-11 as the encoder backbone. The output is
    16x upsampled prediction.
    """

    def __init__(self, num_classes=2, pretrained=False, freeze_pretrained=False):
        super().__init__()

        self.encoder = VGG11Encoder(num_classes, pretrained, freeze_pretrained)
        self.pool4_prediction = nn.Conv2d(512, num_classes, 1)
        self.up2 = UpSample(num_classes, num_classes, 2)
        self.up16 = UpSample(num_classes, num_classes, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img_size = x.shape[2:]

        x = self.encoder.down1(x)  # W/2, H/2
        x = self.encoder.down2(x)  # W/4, H/4
        x = self.encoder.down3(x)  # W/8, H/8
        x = self.encoder.down4(x)  # W/16, H/16
        pool4 = x
        x = self.encoder.down5(x)  # W/32, H/32
        x = self.encoder.prediction(x)

        # skip connection
        pool4_prediction = self.pool4_prediction(pool4)

        x = self.up2(x)  # upsample prediction by 2x
        # resize to size of pool4 prediction, x.shape <= pool4_prediction.shape
        x = F.interpolate(x, pool4_prediction.shape[2:], mode="bilinear")
        x = x + pool4_prediction  # fuse pool4 with prediction

        # upsample 16x to get output
        x = self.up16(x)
        x = F.interpolate(x, img_size, mode="bilinear")

        return x
