import os
from torch.utils.data import Dataset
from torchvision.io import read_image


class ImageSegmentationDataset(Dataset):
    def __init__(
        self, img_dir: str, img_mask_dir: str, transform=None, mask_transform=None
    ):
        self.img_dir = img_dir
        self.img_mask_dir = img_mask_dir
        self.transform = transform
        self.mask_transform = mask_transform

        # sanity check for dataset count
        assert len(os.listdir(self.img_dir)) == len(
            os.listdir(self.img_mask_dir)
        ), "image and image mask count do not match"

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        # assumes that filenames for image and image mask are the same
        img_path = os.path.join(self.img_dir, os.listdir(self.img_dir)[idx])
        img_mask_path = os.path.join(self.img_mask_dir, os.listdir(self.img_dir)[idx])

        img = read_image(img_path)
        img_mask = read_image(img_mask_path)

        if self.transform:
            img = self.transform(img)
        if self.mask_transform:
            img_mask = self.mask_transform(img_mask)

        return img, img_mask
