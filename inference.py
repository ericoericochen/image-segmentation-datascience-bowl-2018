import torch


def get_mask(pred: torch.Tensor) -> torch.Tensor:
    """
    Given the prediction logits from a neural network for semantic segmentation, return the segmentation
    mask

    Args:
        pred: (N, C, H, W)
            - N: batches
            - C: num classes
            - H, W: height, width
    """
    return torch.argmax(pred, dim=1)


if __name__ == "__main__":
    pred = torch.tensor([[[[10]], [[2]]]])
    mask = get_mask(pred)
    target = torch.tensor([[[0]]])

    assert mask.equal(target)
