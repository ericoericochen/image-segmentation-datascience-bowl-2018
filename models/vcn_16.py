import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import DownSample, UpSample


class VCN16(nn.Module):
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

        self.pool4_prediction = nn.Conv2d(512, num_classes, 1)
        self.up2 = UpSample(num_classes, num_classes, 2)
        self.up16 = UpSample(num_classes, num_classes, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img_size = x.shape[2:]

        x = self.down1(x)  # W/2, H/2
        x = self.down2(x)  # W/4, H/4
        x = self.down3(x)  # W/8, H/8
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

        # upsample 16x to get output
        x = self.up16(x)
        x = F.interpolate(x, img_size, mode="bilinear")

        return x


if __name__ == "__main__":
    shapes = [(1, 3, 112, 112), (1, 3, 945, 673), (1, 3, 448, 448)]

    model = VCN16()
    for shape in shapes:
        inp = torch.rand(shape)
        out = model(inp)

        assert out.shape[2:] == inp.shape[2:]
