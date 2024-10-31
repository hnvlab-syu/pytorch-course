import os
import glob
import lightning as L
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image


SEED = 36
L.seed_everything(SEED)
class ImageNetDataModule(L.LightningDataModule):
    def __init__(self, data_dir:str = '../../dataset', batch_size:int=32):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = glob.glob(os.path.join(self.data_dir, '*/*.jpeg'))
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

    def setup(self, stage: str):
        class_data = list(map(lambda path: os.path.basename(os.path.dirname(path)).split('-', 1), self.dataset))
        class_ids, _ = zip(*class_data)
        train_x, val_x, train_y, val_y = train_test_split(self.dataset, class_ids, test_size=0.2, stratify=class_ids)
        val_x, test_x, val_y, test_y = train_test_split(val_x, val_y, test_size=0.5, stratify=val_y)

        self.train_data = [(x, y) for x, y in zip(train_x, train_y)]
        self.val_data = [(x, y) for x, y in zip(val_x, val_y)]
        self.test_data = [(x, y) for x, y in zip(test_x, test_y)]

        if stage == 'fit':
            self.train_dataset = self.train_data
            self.val_dataset = self.val_data
        
        if stage == 'test':
            self.test_dataset = self.test_data

    def _collate_fn(self, batch):
        images, labels = zip(*batch)
        images = [self.transform(Image.open(img).convert('RGB').copy()) for img in images]
        labels = torch.tensor([int(label) for label in labels], dtype=torch.long)
        return torch.stack(images), labels
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self._collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self._collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self._collate_fn)