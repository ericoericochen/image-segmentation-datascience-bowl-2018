import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import UpSample
from .vgg_base import VGG11Encoder


class VCN32(VGG11Encoder):
    """
    Fully Connected Convolutional Neural Net with VGG-11 as the encoder backbone. The output is
    32x upsampled prediction.
    """

    def __init__(self, num_classes=2, pretrained=False, freeze_pretrained=False):
        super().__init__()

        self.encoder = VGG11Encoder(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_pretrained=freeze_pretrained,
        )
        self.up32 = UpSample(num_classes, num_classes, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img_size = x.shape[2:]

        x = self.encoder(x)
        x = self.up32(x)

        # bilinear upsample if dimensions of prediction is smaller than original image size
        x = F.interpolate(
            x,
            size=img_size,
            mode="bilinear",
        )

        return x
