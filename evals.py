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


def mean_accuracy(model: nn.Module, loader: DataLoader, num_classes=2):
    """
    Evaluates mean accuracy of the predictions by model on data from loader.
    Mean Accuracy = (1 / # classes) * sum(pixel accuracy per class)
    """
    model.eval()

    correct = torch.zeros(num_classes)
    pixels = torch.zeros(num_classes) + 1e-6

    with torch.no_grad():
        for X, Y in loader:
            pred = model(X)
            mask = inference.get_mask(pred)

            # loop over each class and count pixels + correctly predicted
            for i in range(num_classes):
                class_mask = Y == i
                num_pixels = class_mask.sum()
                mask_pixels = mask[class_mask]
                correct_pixels = (mask_pixels == i).sum()

                pixels[i] += num_pixels.item()
                correct[i] += correct_pixels.item()

    acc = correct / pixels
    mean_acc = acc.mean()

    return mean_acc.item()


def mean_IU(model: nn.Module, loader: DataLoader, num_classes=2):
    pass


def frequency_weighted_IU(model: nn.Module, loader: DataLoader, num_classes=2):
    pass


if __name__ == "__main__":
    # pixel accuracy
    model = lambda x: torch.tensor([[[[0, 100], [0, 100]], [[100, 0], [100, 0]]]])
    model.eval = lambda: True

    loader = [(torch.rand(1, 1, 2, 2), torch.tensor([[[1, 0], [1, 1]]]))]

    pixel_acc = pixel_accuracy(model, loader)

    assert pixel_acc == 0.75

    # mean accuracy
    mean_acc = mean_accuracy(model, loader)
    assert torch.isclose(torch.tensor(mean_acc), torch.tensor(5 / 6))
