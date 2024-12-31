import os

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import lightning as L

from src.utils import SEED


class PascalVOC2012DataModule(L.LightningDataModule):
    def __init__(self, data_path: str = '../dataset/VOC2012', batch_size: int = 32, mode: str = 'train', num_workers: int = 0):

        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.mode = mode
        self.num_workers = num_workers
        self.image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256))
        ])

    def setup(self, stage: str = None):
        if self.trainer is not None:
            if self.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.batch_size // self.trainer.world_size

        if self.mode == 'train':
            with open(os.path.join(self.data_path, 'ImageSets', 'Segmentation', 'train.txt'), 'r') as f:
                train_list = f.read().splitlines()
            with open(os.path.join(self.data_path, 'ImageSets', 'Segmentation', 'val.txt'), 'r') as f:
                test_list = f.read().splitlines()
            train_list, val_list = train_test_split(train_list, train_size=0.8, shuffle=True, random_state=SEED)

            train_dataset = self._make_dataset(data_list=train_list)
            val_dataset = self._make_dataset(data_list=val_list)
            test_dataset = self._make_dataset(data_list=test_list)
        else:
            pred_data = [Image.open(self.data_path).convert('RGB')]

        if stage == 'fit':
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset

        if stage == 'test':
            self.test_dataset = test_dataset

        if stage == 'predict':
            self.pred_dataset = pred_data

    def _make_dataset(self, data_list: list):
        dataset = []
        for data in data_list:
            image_path = os.path.join(self.data_path, "JPEGImages", f"{data}.jpg")
            image = Image.open(image_path).convert("RGB")
            image = self.image_transform(image)

            seg_image_path = os.path.join(self.data_path, "SegmentationClass", f"{data}.png")
            mask = Image.open(seg_image_path)
            mask = self.mask_transform(mask)
            target = self._preprocess_mask(mask)

            dataset.append((image, target))
        return dataset
    
    def _preprocess_mask(self, mask):
        mask = np.array(mask)
        mask[mask == 255] = 0
        return torch.tensor(mask, dtype=torch.long)

    def _train_collate_fn(self, batch):
        images, targets = zip(*batch)
        images = torch.stack(images)  
        targets = torch.stack(targets)    
        return images, targets

    def _predict_collate_fn(self, batch):
        img = batch[0]
        input_tensor = self.image_transform(img)
        return input_tensor.unsqueeze(0)  

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers, 
            collate_fn=self._train_collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            collate_fn=self._train_collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._train_collate_fn
        )

    def predict_dataloader(self):
        return DataLoader(
            self.pred_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._predict_collate_fn
        )
