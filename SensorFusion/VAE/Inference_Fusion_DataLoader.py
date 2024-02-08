"""
This dataloader is specifically designed for the fusion inference after training image regeneration.

This dataloader is based on Dataloader used for training. But, the difference is that, it will return
a dictionary with rgb and nir images.

"""

# Standard imports
import os
from torch.utils.data import Dataset
from PIL import Image
import random

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


class InferenceDataset(Dataset):
    """
    Class for loading the freiburg forest images

    """

    def __init__(self, folder_nir_path, folder_rgb_path, read_mode, transforms):
        self.folder_rgb_path = folder_rgb_path
        self.folder_nir_path = folder_nir_path
        self.read_mode = read_mode
        self.transforms = transforms
        self.rgb_img_list = []
        self.nir_img_list = []

        # Check for all the files in the rgb and nir image folders.
        for files in os.listdir(self.folder_rgb_path):
            if os.path.isfile(
                os.path.join(self.folder_rgb_path, files)
            ) and not files.startswith("."):
                self.rgb_img_list.append(os.path.join(self.folder_rgb_path, files))
                self.nir_img_list.append(os.path.join(self.folder_nir_path, files))

        # Check if the same number of images are present in each list.
        assert len(self.rgb_img_list) == len(self.nir_img_list)

    def __len__(self):
        return len(self.rgb_img_list)

    def __getitem__(self, index):
        # Get rgb image path
        rgb_img_path = self.rgb_img_list[index]

        # Get image name
        img_name = rgb_img_path.split("/")[-1]

        # Get nir image path
        nir_img_path = os.path.join(self.folder_nir_path, img_name)

        # Read images
        if self.read_mode == "RGB":
            rgb_img = Image.open(rgb_img_path).convert("RGB")
            nir_img = Image.open(nir_img_path).convert("RGB")

        elif self.read_mode == "Gray":
            rgb_img = Image.open(rgb_img_path).convert("L")
            nir_img = Image.open(nir_img_path).convert("L")

        else:
            raise ValueError(
                f"{self.read_mode} is not a valid value for the argument 'read_mode'."
            )

        # Check and apply for transforms
        if self.transforms:
            rgb_img = self.transforms(rgb_img)
            nir_img = self.transforms(nir_img)

        sample = {"rgb_img": rgb_img, "nir_img": nir_img}

        return sample
