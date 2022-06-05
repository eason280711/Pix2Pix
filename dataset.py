import numpy as np
import os

import torch.utils.data as data

import albumentations
from albumentations.pytorch import ToTensorV2

from PIL import Image

class dataset(data.Dataset):
    def __init__(self,img_dir):
        self.img_dir = img_dir
        self.list_files = os.listdir(self.img_dir)
        self.both_transform = albumentations.Compose(
            [
                albumentations.Resize(width=256,height=256)
            ],
            additional_targets={"image0":"image"}
        )

        self.input_transform = albumentations.Compose(
            [
                albumentations.HorizontalFlip(p=0.5),
                albumentations.Normalize(max_pixel_value=255.0),
                ToTensorV2()
            ]
        )

        self.target_transform = albumentations.Compose(
            [
                albumentations.Normalize(max_pixel_value=255.0),
                ToTensorV2()
            ]
        )

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.img_dir,img_file)
        image = np.array(Image.open(img_path))
        w = len(image[0]) // 2
        input_image = image[:,w:,:]
        target_image = image[:,:w,:]


        augmentations = self.both_transform(image=input_image,image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = self.input_transform(image=input_image)["image"]
        target_image = self.target_transform(image=target_image)["image"]

        return input_image,target_image