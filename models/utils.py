import torch
import torch.nn as nn


class DownSample(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, layers: int, scale: int = 2
    ):
        super().__init__()

        assert layers >= 1

        modules = []

        for i in range(layers):
            inp_channels = in_channels if i == 0 else out_channels
            conv = nn.Conv2d(inp_channels, out_channels, kernel_size=3, padding=1)
            relu = nn.ReLU()

            modules.append(conv)
            modules.append(relu)

        pool = nn.MaxPool2d(scale, scale)
        modules.append(pool)

        self.conv_pool = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_pool(x)


class UpSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale: int):
        super().__init__()
        self.conv_tranpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2 * scale,
            stride=scale,
            padding=int(scale / 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_tranpose(x)


def flatten_module(module: nn.Module) -> list[nn.Module]:
    """
    Returns a list of the primitive modules in module
    """
    children = [child for child in module.children()]

    if len(children) == 0:
        return [module]

    return flatten_modules(children)


def flatten_modules(modules: list[nn.Module]) -> list[nn.Module]:
    """
    Flattens the modules and returns a list of the primitive modules used in network
    """
    flattened = []

    for module in modules:
        flattened_modules = flatten_module(module)

        for mod in flattened_modules:
            flattened.append(mod)

    return flattened
