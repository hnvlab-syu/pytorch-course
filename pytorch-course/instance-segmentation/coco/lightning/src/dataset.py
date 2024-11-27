import os
import torch
import numpy as np
import lightning as L
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms 
from pycocotools.coco import COCO
from PIL import Image
from sklearn.model_selection import train_test_split


class COCODataModule(L.LightningDataModule):
    def __init__(self, data_path: str, annotation_file: str, batch_size: int = 16, mode: str = 'train'):
        super().__init__()
        self.data_path = data_path
        self.annotation_file = annotation_file
        self.batch_size = batch_size
        self.mode = mode
        self.coco = None
        self.batch_size_per_device = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.ColorJitter(brightness=0.5),
        ])
    
    def class_names(self):
        if self.coco is None:
            self.coco = COCO(self.annotation_file)

        categories = self.coco.loadCats(self.coco.getCatIds())
        max_category_id = max(cat['id'] for cat in categories)
        class_names = [''] * (max_category_id + 1)

        for category in categories:
            class_names[category['id']] = category['name']
            
        return class_names

    def setup(self, stage=None):
        if self.trainer is not None:
            if self.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.batch_size // self.trainer.world_size

        if self.mode == 'train':
            coco = COCO(self.annotation_file)
            image_ids = list(coco.imgs.keys())
        
            train_val_ids, test_ids = train_test_split(image_ids, test_size=0.2, random_state=42)
            train_ids, val_ids = train_test_split(train_val_ids, test_size=0.2, random_state=42)
            
            self.train_dataset = COCODataset(self.data_path, self.annotation_file, 
                                             image_ids=train_ids, transform=self.transform)
            self.val_dataset = COCODataset(self.data_path, self.annotation_file, 
                                           image_ids=val_ids, transform=self.transform)
            self.test_dataset = COCODataset(self.data_path, self.annotation_file, 
                                            image_ids=test_ids, transform=self.transform)
        else:
            self.pred_dataset = [Image.open(self.data_path).convert('RGB')]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size_per_device,
            shuffle=True,
            collate_fn=self._train_collate_fn,
            num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            collate_fn=self._train_collate_fn,
            num_workers=4
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            collate_fn=self._train_collate_fn,
            num_workers=4
        )

    def predict_dataloader(self):
        return DataLoader(
            self.pred_dataset,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            collate_fn=self._predict_collate_fn
        )

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
        input = self.transform(img).unsqueeze(0)
        return input, np.array(img)
        
class COCODataset(Dataset):
    def __init__(self, image_path, annotation_file, image_ids=None, transform=None):
        self.coco = COCO(annotation_file)
        self.image_folder = image_path
        self.image_ids = image_ids if image_ids is not None else list(self.coco.imgs.keys())
    
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_ids)
        
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = f"{self.image_folder}/{image_info['file_name']}"
        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
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
                # Convert to float32 and normalize to [0, 1]
                mask_tensor = torch.tensor(np.array(resized_mask), dtype=torch.float32)
                mask_tensor = mask_tensor / 255.0 if mask_tensor.max() > 1.0 else mask_tensor
                resized_masks.append(mask_tensor)
            
            masks = torch.stack(resized_masks)

        if self.transform:
            image = self.transform(image)
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
            "image_id": torch.tensor([image_id])
        }

        

        return image, target
    

