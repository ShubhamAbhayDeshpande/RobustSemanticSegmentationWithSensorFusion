"""
The following will be a dataloader for the freiburg forest images. 

The need for the dataloader is that, the default dataloader is for loading images from various folders. 
This structure is not suitable for our application hence we need to write a different dataloader.

The inspiration for the dataloader is taken from the official pytorch documentation.

https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

"""

# Standard imports
import os
from torch.utils.data import Dataset
from PIL import Image
import random

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


class ForestDataset_RGB(Dataset):
    """
    Class for loading only the RGB images in the training or inference model.

    """

    def __init__(self, folder_path, transforms):
        self.folder_path = folder_path
        self.transforms = transforms
        self.img_lst = []

        for files in os.listdir(self.folder_path):
            if os.path.isfile(
                os.path.join(self.folder_path, files)
            ) and not files.startswith("."):
                self.img_lst.append(os.path.join(self.folder_path, files))

        random.shuffle(self.img_lst)

    def __len__(self):
        return len(self.img_lst)

    def __getitem__(self, index):
        # Get rgb image path
        rgb_img_path = self.img_lst[index]

        # Get image name
        img_name = rgb_img_path.split("/")[-1]

        # Check if image exists, if not raise an error.
        if not os.path.isfile(rgb_img_path):
            raise FileExistsError(f"The image {rgb_img_path} does not exist")

        # Read images
        rgb_img = Image.open(rgb_img_path).convert("RGB")

        # Check and apply for transforms
        if self.transforms:
            rgb_img = self.transforms(rgb_img)

        sample = {"rgb_img": rgb_img}

        return sample


class ForestDataset_RGB_NIR(Dataset):
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

        self.final_list = self.rgb_img_list + self.nir_img_list
        random.shuffle(self.final_list)

    def __len__(self):
        return len(self.final_list)

    def __getitem__(self, index):
        img_path = self.final_list[index]

        if not os.path.isfile(img_path):
            raise FileExistsError(f"No file named {img_path}")

        if self.read_mode == "RGB":
            img = Image.open(img_path).convert("RGB")

        elif self.read_mode == "Gray":
            img = Image.open(img_path).convert("L")

        else:
            raise ValueError(
                f"{self.read_mode} is not a valid value for the argument 'read_mode'."
            )

        # Check if transforms are applicable and apply them
        if self.transforms:
            img = self.transforms(img)

        sample = {"img": (img).float()}

        return sample


class InferenceDataset_RGB_NIR(Dataset):
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
        if not os.path.isfile(nir_img_path):
            raise FileExistsError(f"The image {nir_img_path} does not exist")

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
