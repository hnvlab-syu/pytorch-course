import os

from PIL import Image

import torch

from src.utils import preprocess_mask


class PascalVOCDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, image_list, image_transform=None, mask_transform=None):
        self.data_dir = data_dir
        self.image_list = image_list
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        image_name = self.image_list[idx]

        img_path = os.path.join(self.data_dir, 'JPEGImages', f'{image_name}.jpg')
        image = Image.open(img_path).convert("RGB")
        image = self.image_transform(image)

        mask_path = os.path.join(self.data_dir, 'SegmentationClass', f'{image_name}.png')
        mask = Image.open(mask_path)
        mask = self.mask_transform(mask)
        mask = preprocess_mask(mask)

        return image, mask
    