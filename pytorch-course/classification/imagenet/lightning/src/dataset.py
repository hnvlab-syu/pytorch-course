import os
import glob

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import lightning as L


SEED = 36
L.seed_everything(SEED)


class ImageNetDataModule(L.LightningDataModule):
    def __init__(self, data_path: str = '../../dataset', batch_size: int = 32, mode: str = 'train'):
        super().__init__()
        self.mode = mode
        if self.mode == 'train':
            self.dataset = glob.glob(os.path.join(data_path, '*/*.jpeg'))
        else:
            self.data_path = data_path
            batch_size = 1

        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

    def setup(self, stage: str):
        if self.trainer is not None:
            if self.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.batch_size // self.trainer.world_size

        if self.mode == 'train':
            class_data = list(map(lambda path: os.path.basename(os.path.dirname(path)).split('-', 1), self.dataset))
            class_ids, _ = zip(*class_data)
            train_x, val_x, train_y, val_y = train_test_split(self.dataset, class_ids, test_size=0.2, stratify=class_ids)
            val_x, test_x, val_y, test_y = train_test_split(val_x, val_y, test_size=0.5, stratify=val_y)

            train_data = [(x, y) for x, y in zip(train_x, train_y)]
            val_data = [(x, y) for x, y in zip(val_x, val_y)]
            test_data = [(x, y) for x, y in zip(test_x, test_y)]
        else:
            pred_data = [Image.open(self.data_path).convert('RGB')]

        if stage == 'fit':
            self.train_dataset = train_data
            self.val_dataset = val_data
        
        if stage == 'test':
            self.test_dataset = test_data

        if stage == 'predict':
            self.pred_dataset = pred_data

    def _train_collate_fn(self, batch):
        images, labels = zip(*batch)
        images = [self.transform(Image.open(img).convert('RGB')) for img in images]
        labels = torch.tensor([int(label) for label in labels], dtype=torch.long)
        return torch.stack(images), labels
    
    def _predict_collate_fn(self, batch):
        img = batch[0]
        input = self.transform(img).unsqueeze(0)
        return input, np.array(img)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size_per_device, shuffle=True, collate_fn=self._train_collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size_per_device, shuffle=False, collate_fn=self._train_collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size_per_device, shuffle=False, collate_fn=self._train_collate_fn)
    
    def predict_dataloader(self):
        return DataLoader(self.pred_dataset, batch_size=self.batch_size_per_device, shuffle=False, collate_fn=self._predict_collate_fn) 