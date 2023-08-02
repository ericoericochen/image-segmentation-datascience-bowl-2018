import os
from torch.utils.data import Dataset
from PIL import Image, ImageChops


class SegmentationDataset(Dataset):
    def __init__(self, root: str, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.image_ids = os.listdir(self.root)

    def _combine_masks(self, masks: list[Image.Image]) -> Image.Image:
        if len(masks) == 1:
            return masks[0]

        mask = ImageChops.add(masks[0], masks[1])
        for msk in masks[2:]:
            mask = ImageChops.add(mask, msk)

        return mask

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        # get image id
        image_id = self.image_ids[idx]

        # get path to image and masks
        image_path = os.path.join(self.root, image_id)
        masks_path = os.path.join(image_path, "masks")

        # get image
        image_path = os.path.join(image_path, "images")
        image_filename = os.listdir(image_path)[0]
        image_path = os.path.join(image_path, image_filename)
        image = Image.open(image_path)

        # combine masks into one
        mask_filenames = os.listdir(masks_path)
        masks = [Image.open(os.path.join(masks_path, mask_filename)).convert("L")
                 for mask_filename in mask_filenames]  # masks are grayscale

        mask = self._combine_masks(masks)

        if self.transform:
            image = self.transform(image)

        return image, mask
