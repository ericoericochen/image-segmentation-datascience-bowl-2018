import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from .utils import DownSample, UpSample, flatten_modules


class VCN8(nn.Module):
    """
    Fully Connected Convolutional Neural Net with VGG-11 as the encoder backbone. The output is 
    8x upsampled prediction.
    """

    def __init__(self, num_classes=2, pretrained=False, freeze_pretrained=False):
        super().__init__()
        self.pretrained = pretrained
        self.freeze_pretrained = freeze_pretrained

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

        self.pool3_prediction = nn.Conv2d(256, num_classes, 1)
        self.pool4_prediction = nn.Conv2d(512, num_classes, 1)

        self.up2 = UpSample(num_classes, num_classes, 2)
        self.up2pool = UpSample(num_classes, num_classes, 2)
        self.up8 = UpSample(num_classes, num_classes, 8)

        self.initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img_size = x.shape[2:]

        x = self.down1(x)  # W/2, H/2
        x = self.down2(x)  # W/4, H/4
        x = self.down3(x)  # W/8, H/8
        pool3 = x
        x = self.down4(x)  # W/16, H/16
        pool4 = x
        x = self.down5(x)  # W/32, H/32
        x = self.prediction(x)

        # skip connection
        pool4_prediction = self.pool4_prediction(pool4)

        x = self.up2(x)  # upsample prediction by 2x
        # resize to size of pool4 prediction, x.shape <= pool4_prediction.shape
        x = F.interpolate(x, pool4_prediction.shape[2:], mode="bilinear")
        x = x + pool4_prediction  # fuse pool4 with prediction

        x = self.up2pool(x)

        pool3_prediction = self.pool3_prediction(pool3)
        x = F.interpolate(x, pool3_prediction.shape[2:], mode="bilinear")
        x = x + pool3_prediction

        # upsample 8x to get output
        x = self.up8(x)
        x = F.interpolate(x, img_size, mode="bilinear")

        return x

    def initialize_weights(self):
        if self.pretrained:
            print("PRETRAINED")
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

                    if self.freeze_pretrained:
                        layer.requires_grad_(False)
