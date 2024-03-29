import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import inference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def loss(model: nn.Module, criterion: nn.Module, loader: DataLoader, device=device) -> float:
    """
    Computes loss over entire loader
    """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for X, Y in loader:
            X = X.to(device)
            Y = Y.to(device)

            pred = model(X)
            loss = criterion(pred, Y)
            total_loss += loss.item()

    N = len(loader)
    loss = total_loss / N

    return loss


def pixel_accuracy(model: nn.Module, loader: DataLoader, device=device) -> float:
    """
    Evaluates pixel accuracy of the predictions by model on data from loader.
    Pixel Accuracy = # correctly predicted pixels / # total pixels
    """
    model.eval()

    num_correctly_predicted = 0
    num_pixels = 0

    with torch.no_grad():
        for X, Y in loader:
            X = X.to(device)
            Y = Y.to(device)

            pred = model(X)
            mask = inference.get_mask(pred)
            eqs = mask == Y

            correctly_predicted = eqs.sum()
            batch_pixels = Y.size(0) * Y.size(1) * Y.size(2)

            num_correctly_predicted += correctly_predicted.item()
            num_pixels += batch_pixels

    acc = num_correctly_predicted / num_pixels

    return acc


def mean_accuracy(model: nn.Module, loader: DataLoader, device=device, num_classes=2):
    """
    Evaluates mean accuracy of the predictions by model on data from loader.
    Mean Accuracy = (1 / # classes) * sum(pixel accuracy per class)
    """
    model.eval()

    correct = torch.zeros(num_classes)
    pixels = torch.zeros(num_classes) + 1e-6

    with torch.no_grad():
        for X, Y in loader:
            X = X.to(device)
            Y = Y.to(device)

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


def mean_IU(model: nn.Module, loader: DataLoader, device=device, num_classes=2):
    """
    Evaluates mean IU of the predictions by model on data from loader.
    Mean IU = (1 / # classes) * (sum IU per class)
    """
    model.eval()

    intersection = torch.zeros(num_classes)
    union = torch.zeros(num_classes) + 1e-6

    with torch.no_grad():
        for X, Y in loader:
            X = X.to(device)
            Y = Y.to(device)

            pred = model(X)
            mask = inference.get_mask(pred)

            for i in range(num_classes):
                class_mask = Y == i
                n_ii = (mask[class_mask] == i).sum()  # TP
                t_i = class_mask.sum()  # TP + FN
                sum_n_ji = (mask == i).sum()  # FP + TP

                intersection[i] = n_ii
                union[i] = t_i + sum_n_ji - n_ii

    iu = intersection / union
    mean_iu = iu.mean()

    return mean_iu.item()


def frequency_weighted_IU(model: nn.Module, loader: DataLoader, device=device, num_classes=2):
    """
    Evaluates frequency weighted  IU of the predictions by model on data from loader.
    Frequency Weighted IU = (1 / # pixels) * sum(IOU per class * pixels per class)
    """
    model.eval()

    sum_ti = 0
    intersection = torch.zeros(num_classes)
    union = torch.zeros(num_classes) + 1e-6

    with torch.no_grad():
        for X, Y in loader:
            X = X.to(device)
            Y = Y.to(device)

            pred = model(X)
            mask = inference.get_mask(pred)

            for i in range(num_classes):
                class_mask = Y == i
                n_ii = (mask[class_mask] == i).sum()  # TP
                t_i = class_mask.sum()  # TP + FN
                sum_n_ji = (mask == i).sum()  # FP + TP

                intersection[i] = n_ii * t_i
                union[i] = t_i + sum_n_ji - n_ii

                sum_ti += t_i.item()

    weighted_iu = intersection / union
    frequency_weighted_iu = (1 / sum_ti) * weighted_iu.sum()

    return frequency_weighted_iu.item()


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

    # mean IU
    mean_iu = mean_IU(model, loader)
    assert torch.isclose(torch.tensor(mean_iu), torch.tensor(7 / 12))

    # frequency weighted IU
    frequency_mean_iu = frequency_weighted_IU(model, loader)
    assert torch.isclose(torch.tensor(frequency_mean_iu), torch.tensor(5 / 8))
