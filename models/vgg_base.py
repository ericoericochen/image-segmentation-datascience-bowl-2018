import torch
import torch.nn as nn
import torchvision
from .utils import DownSample, flatten_modules


class VGG11Base(nn.Module):
    """
    Convolutional Layers of VGG 11
    """

    def __init__(self, pretrained=False):
        self.pretrained = pretrained
        self.down1 = DownSample(3, 64, 1)
        self.down2 = DownSample(64, 128, 1)
        self.down3 = DownSample(128, 256, 2)
        self.down4 = DownSample(256, 512, 2)
        self.down5 = DownSample(512, 512, 2)

    def forward(self, x: torch.Tensor):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)

        return x

    def _initialize_weights(self):
        if self.pretrained:
            # get pretrained model
            pretrained = torchvision.models.vgg11(weights="IMAGENET1K_V1")

            # get convolutional layers of pretrained model
            pretrained_layers = [layer for layer in pretrained.features.children()]

            # get layers of current network
            layers = flatten_modules(
                [self.down1, self.down2, self.down3, self.down4, self.down5]
            )

            assert len(pretrained_layers) == len(layers)

            # get convolutional layers of pretrained model and copy the weights to FCN
            for layer, pretrained_layer in zip(layers, pretrained_layers):
                if isinstance(layer, nn.Conv2d) and isinstance(
                    pretrained_layer, nn.Conv2d
                ):
                    assert layer.weight.size() == pretrained_layer.weight.size()
                    assert layer.bias.size() == pretrained_layer.bias.size()

                    layer.weight.data.copy_(pretrained_layer.weight.data)
                    layer.bias.data.copy_(pretrained_layer.bias.data)
