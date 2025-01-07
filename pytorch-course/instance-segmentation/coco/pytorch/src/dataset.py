import os

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO


class COCODataset(Dataset):
    def __init__(self, image_dir, annotation_file, image_list, transform):
        self.coco = COCO(annotation_file)
        self.image_dir = image_dir
        self.image_list = image_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image_path = self.image_list[index]
        file_name = os.path.basename(image_path)

        for img in self.coco.dataset['images']:
            if img['file_name'] == file_name:
                image_id = img['id']

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
