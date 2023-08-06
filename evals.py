import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import inference


def pixel_accuracy(model: nn.Module, loader: DataLoader) -> float:
    """
    Evaluates pixel accuracy of the predictions by model on data from loader.
    Pixel Accuracy = # correctly predicted pixels / # total pixels
    """
    model.eval()

    num_correctly_predicted = 0
    num_pixels = 0

    with torch.no_grad():
        for X, Y in loader:
            pred = model(X)
            mask = inference.get_mask(pred)
            eqs = mask == Y

            correctly_predicted = eqs.sum()
            batch_pixels = Y.size(1) * Y.size(2)

            num_correctly_predicted += correctly_predicted.item()
            num_pixels += batch_pixels

    acc = num_correctly_predicted / num_pixels

    return acc


if __name__ == "__main__":
    model = lambda x: torch.tensor([[[[0, 100], [0, 0]], [[100, 0], [100, 100]]]])
    model.eval = lambda: True

    loader = [(torch.rand(1, 1, 2, 2), torch.tensor([[[1, 1], [1, 1]]]))]

    pixel_acc = pixel_accuracy(model, loader)

    assert pixel_acc == 0.75
