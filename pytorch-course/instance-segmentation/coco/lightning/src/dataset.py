import os

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from pycocotools.coco import COCO

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import lightning as L

from src.utils import SEED


class COCODataModule(L.LightningDataModule):
    def __init__(self, data, batch_size=8, mode='train', num_workers=0):
        super().__init__()
        if mode == 'train':
            self.data_path = os.path.join(data, 'val2017')
            self.coco = COCO(os.path.join(data, 'instances_val2017.json'))
        elif mode == 'predict':
            self.data_path = os.path.join(data)
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
        self.pred_transform = transforms.Compose([
            transforms.ToTensor()
        ])
    
    def setup(self, stage=None):
        if self.trainer is not None:
            if self.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.batch_size // self.trainer.world_size

        if self.mode == 'train':
            image_ids = list(self.coco.imgs.keys())
            train_ids, val_ids = train_test_split(image_ids, test_size=0.2, random_state=SEED)
            val_ids, test_ids = train_test_split(val_ids, test_size=0.5, random_state=SEED)
            
            print('Make datasets')
            train_dataset = self._make_dataset(data_ids=train_ids)
            val_dataset = self._make_dataset(data_ids=val_ids)
            test_dataset = self._make_dataset(data_ids=test_ids)
            print('Done')
        
        else:
            pred_dataset = [Image.open(self.data_path).convert('RGB')]

        if stage == 'fit':
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset

        if stage == 'test':
            self.test_dataset = test_dataset

        if stage == 'predict':
            self.pred_dataset = pred_dataset

    def _make_dataset(self, data_ids):
        dataset = []
        for data in data_ids:
            image_path = os.path.join(self.data_path, f"{str(data).zfill(12)}.jpg")
            image = Image.open(image_path).convert("RGB")
            original_width, original_height = image.size

            ann_ids = self.coco.getAnnIds(imgIds=data)
            annotations = self.coco.loadAnns(ann_ids)
            
            boxes = []
            labels = []
            raw_masks = []

            for ann in annotations:
                x, y, w, h = ann['bbox']
                boxes.append([x, y, x + w, y + h])
                labels.append(ann['category_id'])
                mask = self.coco.annToMask(ann)
                raw_masks.append(mask)
            
            if len(boxes) == 0:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
                masks = torch.zeros((0, 256, 256), dtype=torch.float32)
            else:
                boxes = torch.tensor(boxes, dtype=torch.float32)
                labels = torch.tensor(labels, dtype=torch.int64)

                resized_masks = []
                for mask in raw_masks:
                    mask_image = Image.fromarray(mask)
                    resized_mask = mask_image.resize((256, 256), resample=Image.NEAREST)
                    mask_tensor = torch.tensor(np.array(resized_mask), dtype=torch.float32)
                    mask_tensor = mask_tensor / 255.0 if mask_tensor.max() > 1.0 else mask_tensor
                    resized_masks.append(mask_tensor)
                
                masks = torch.stack(resized_masks)

            image = self.image_transform(image)            
            new_width, new_height = 256, 256

            scale_x = new_width / original_width
            scale_y = new_height / original_height

            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, new_width)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, new_height)

            target = {
                "boxes": boxes,
                "labels": labels,
                "masks": masks,
                "image_id": torch.tensor([data])
            }

            dataset.append((image, target))

        return dataset
    
    def _train_collate_fn(self, batch):
        images = []
        targets = []
        
        for image, target in batch:
            images.append(image)
            targets.append(target)
        
        images = torch.stack(images, dim=0)
        
        return images, targets

    def _predict_collate_fn(self, batch):
        img = batch[0]
        input = self.pred_transform(img).unsqueeze(0)
        
        return input, np.array(img)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size_per_device,
            shuffle=True,
            collate_fn=self._train_collate_fn,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            collate_fn=self._train_collate_fn,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            collate_fn=self._train_collate_fn,
            num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self.pred_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=self._predict_collate_fn
        )
