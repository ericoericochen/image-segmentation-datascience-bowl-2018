import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from .utils import DownSample, UpSample,  flatten_modules


class VCN32(nn.Module):
    """
    Fully Connected Convolutional Neural Net with VGG-11 as the encoder backbone. The output is 
    32x upsampled prediction.
    """

    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()
        self.pretrained = pretrained
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

        self.initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def initialize_weights(self):
        if self.pretrained:
            # get pretrained model
            pretrained = torchvision.models.vgg11(weights="IMAGENET1K_V1")

            # get convolutional layers of pretrained model
            pretrained_layers = [
                layer for layer in pretrained.features.children()]

            # get layers of current network
            layers = flatten_modules(
                [self.down1, self.down2, self.down3, self.down4, self.down5])

            assert len(pretrained_layers) == len(layers)

            # get convolutional layers of pretrained model and copy the weights to FCN
            for layer, pretrained_layer in zip(layers, pretrained_layers):
                if isinstance(layer, nn.Conv2d) and isinstance(pretrained_layer, nn.Conv2d):
                    assert layer.weight.size() == pretrained_layer.weight.size()
                    assert layer.bias.size() == pretrained_layer.bias.size()

                    layer.weight.data.copy_(pretrained_layer.weight.data)
                    layer.bias.data.copy_(pretrained_layer.bias.data)
