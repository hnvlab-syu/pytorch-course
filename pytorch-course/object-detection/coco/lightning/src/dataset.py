import os
import json
import glob
import torch
import numpy as np
import lightning as pl
import albumentations as A

from PIL import Image
from torch.utils.data import DataLoader


class COCODataModule(pl.LightningDataModule):
    def __init__(self, 
        datasets_path, 
        annots_path, 
        batch_size,
        num_workers):
        super().__init__()
        self.datasets_path = datasets_path
        self.annots_path = annots_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transforms = A.Compose([
                                A.HorizontalFlip(p=0.5),
                                A.RandomBrightnessContrast(p=0.2),
                            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

        self.else_transforms = A.Compose([      # val, test, pred
                            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    # def prepare_data(self):
    #     # download

    def setup(self, stage: str):
        with open(os.path.join(self.annots_path, "captions_train2017"), 'r') as f:
            self.train_annots = json.load(f)

        with open(os.path.join(self.annots_path, "captions_val2017"), 'r') as f:
            self.val_annots = json.load(f)
            
        if stage == "fit":
            self.train_data = []
            for img in self.train_annots['images']:
                img_id = img['id']
                img_path = os.path.join(self.datasets_path, "train2017", img['file_name'])  # (ex) file_name: 000000391895.jpg (확장자까지 포함)
                
                annots = [x for x in self.train_annots['annotations']
                            if x['image_id'] == img_id]
                
                self.train_data.append({
                    'image_path': img_path,
                    'boxes': [x['bbox'] for x in annots],
                    'labels': [x['category_id'] for x in annots]
                })
            
            self.val_data = []
            for img in self.val_annots['images']:
                img_id = img['id']
                img_path = os.path.join(self.datasets_path, "val2017", img['file_name'])
                
                annots = [x for x in self.val_annots['annotations'] 
                            if x['image_id'] == img_id]
                
                self.val_data.append({
                    'image_path': img_path,
                    'boxes': [x['bbox'] for x in annots],
                    'labels': [x['category_id'] for x in annots]
                })

            print('---------------------')
            print(f"Train size: {len(self.train_data)}, Val size: {len(self.val_data)}")

        if stage == "test":
            self.test_data = sorted(glob.glob(os.path.join(self.datasets_path, "test2017", "*.{jpg, jpeg, png}")))

            print('---------------------')
            print(f"Test size: {len(self.test_data)}")

        if stage == "predict":
            self.pred_data = sorted(glob.glob(os.path.join(self.datasets_path, "unlabeled2017", "*.{jpg, jpeg, png}")))

            print('---------------------')
            print(f"Predict size: {len(self.pred_data)}")    


    def _collate_fn(self, batch):
        images = []
        targets = []
        
        for b in batch:     # setup 처리 후의 train_data etc.
            img = Image.open(b['image_path']).convert('RGB')
            img = np.array(img)
            
            # bbox 포맷 (COCO -> PASCAL_VOC)
            boxes = np.array(b['boxes'], dtype=np.float32)
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
            
            labels = np.array(b['labels'], dtype=np.int64)
        
            # augmentation
            if self.train_transforms:   
                transformed = self.train_transforms(image=img, bboxes=boxes, class_labels=labels)
                img = transformed['image']
                boxes = transformed['bboxes']
                labels = transformed['class_labels']

            elif self.else_transforms:      # val, test, pred
                transformed = self.else_transforms(image=img, bboxes=boxes, class_labels=labels)
                img = transformed['image']
                boxes = transformed['bboxes']
                labels = transformed['class_labels']

            img = img.transpose(2, 0, 1)  # HWC -> CHW
            img = img / 255.0  # [0, 1] 정규화
            img = torch.Tensor(img)
            
            images.append(img)
            targets.append({
                'boxes': torch.as_tensor(boxes, dtype=torch.float32),
                'labels': torch.as_tensor(labels, dtype=torch.int64)
            })
            
        return images, targets


    def train_dataloader(self):
        return DataLoader(self.train_data, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          collate_fn=self._train_collate_fn, 
                          shuffle=True, 
                          drop_last=True, 
                          pin_memory=True)
        
    def val_dataloader(self):
        return DataLoader(self.val_data, 
                          batch_size=int(self.batch_size/4), 
                          num_workers=self.num_workers, 
                          collate_fn=self._else_collate_fn, 
                          shuffle=False, 
                          drop_last=True, 
                          pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, 
                          batch_size=int(self.batch_size/4), 
                          num_workers=self.num_workers, 
                          collate_fn=self._else_collate_fn, 
                          shuffle=False, 
                          drop_last=True, 
                          pin_memory=True)

    def predict_dataloader(self):
        return DataLoader(self.pred_data, 
                          batch_size=int(self.batch_size), 
                          num_workers=self.num_workers, 
                          collate_fn=self._else_collate_fn, 
                          shuffle=False, 
                          drop_last=True, 
                          pin_memory=True)
        

   
    