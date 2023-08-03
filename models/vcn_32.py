import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import DownSample, UpSample


class VCN32(nn.Module):
    """
    Fully Connected Convolutional Neural Net with VGG-11 as the encoder backbone. The output is 
    32x upsampled prediction.
    """

    def __init__(self, num_classes=2):
        super().__init__()
        self.down1 = DownSample(3, 64, 1)
        self.down2 = DownSample(64, 128, 1)
        self.down3 = DownSample(128, 256, 2)
        self.down4 = DownSample(256, 512, 2)
        self.down5 = DownSample(512, 512, 2)
        self.prediction = nn.Sequential(
            nn.Conv2d(512, 4096, 3, 1, 1),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 3, 1, 1),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Conv2d(4096, num_classes, 1, 1)
        )
        self.up32 = UpSample(num_classes, num_classes, 32)

    def forward(self, x: torch.Tensor):
        img_size = x.shape[2:]

        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        x = self.prediction(x)
        x = self.up32(x)

        # bilinear upsample if dimensions of prediction is smaller than original image size
        x = F.interpolate(x, size=img_size, mode="bilinear")

        return x
