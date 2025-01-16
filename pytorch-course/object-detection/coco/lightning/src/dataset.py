# COCO validatdion dataset 8:1:1로 나눠서 사용
import os
import json
import torch
import numpy as np
import lightning as L
import albumentations as A

from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


SEED = 36
L.seed_everything(SEED)

class COCODataModule(L.LightningDataModule):
    def __init__(self, 
        data_path, 
        annots_path, 
        batch_size,
        num_workers,
        mode):
        super().__init__()
        self.data_path = data_path
        self.annots_path = annots_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mode = mode
        self.train_transforms = A.Compose([
                                A.HorizontalFlip(p=0.5),
                                A.RandomBrightnessContrast(p=0.2),
                            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        self.val_transforms = A.Compose([
                            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    def setup(self, stage: str):
        if self.mode != 'predict':
            with open(os.path.join(self.annots_path, "instances_val2017.json"), 'r') as f:
                self.val_annots = json.load(f)
    
            data = []
            print('Loading annotations...')
            for img in tqdm(self.val_annots['images']):
                img_id = img['id']
                img_path = os.path.join(self.data_path, img['file_name'])
                annots = [x for x in self.val_annots['annotations'] if x['image_id'] == img_id]
                
                if annots:
                    data.append({
                        'image_path': img_path,
                        'boxes': [x['bbox'] for x in annots],
                        'labels': [x['category_id'] for x in annots]
                    })

        if self.mode == 'train':
            train_data, temp_data = train_test_split(data, test_size=0.2, random_state=SEED)
            val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=SEED)

        if stage == "fit":
            self.train_dataset = train_data
            self.val_dataset = val_data
            print('---------------------')
            print(f"Train size: {len(self.train_dataset)}, Val size: {len(self.val_dataset)}")

        if stage == "test":
            self.test_dataset = test_data
            print('---------------------')
            print(f"Test size: {len(self.test_dataset)}")

        if stage == "predict":
            self.pred_dataset = [self.data_path]  


    def _train_collate_fn(self, batch):
        images = []
        targets = []
        
        for b in batch:
            img = Image.open(b['image_path']).convert('RGB')
            img = np.array(img)
            
            boxes = np.array(b['boxes'], dtype=np.float32)  # bbox 형식 변환: [x, y, width, height]-> [x1, y1, x2, y2] 좌/우상단
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
            
            labels = np.array(b['labels'], dtype=np.int64)
        
            transformed = self.train_transforms(image=img, bboxes=boxes, class_labels=labels)
            
            img = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['class_labels']

            img = img.transpose(2, 0, 1)
            img = img / 255.0
            img = torch.Tensor(img)
            
            images.append(img)
            targets.append({
                'boxes': torch.tensor(boxes, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.int64)
            })
    
        return images, targets

    def _val_test_collate_fn(self, batch):
        images = []
        targets = []
        
        for b in batch:
            img = Image.open(b['image_path']).convert('RGB')
            img = np.array(img)
            
            boxes = np.array(b['boxes'], dtype=np.float32) 
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
            
            labels = np.array(b['labels'], dtype=np.int64)
        
            transformed = self.val_transforms(image=img, bboxes=boxes, class_labels=labels)

            img = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['class_labels']

            img = img.transpose(2, 0, 1)
            img = img / 255.0
            img = torch.Tensor(img)
            
            images.append(img)
            targets.append({
                'boxes': torch.tensor(boxes, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.int64)
            })
    
        return images, targets
    
    def _pred_collate_fn(self, batch):
        images = []
        
        for b in batch:
            img_path = b  
            img = Image.open(img_path).convert('RGB')
            img = np.array(img)
            
            transformed = self.val_transforms(image=img)
            img = transformed['image']

            img = torch.tensor(img).permute(2, 0, 1)  
            img = img.float() / 255.0
            
            images.append(img)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          collate_fn=self._train_collate_fn, 
                          shuffle=True, 
                          drop_last=True, 
                          pin_memory=True)
        
    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=int(self.batch_size/4), 
                          num_workers=self.num_workers, 
                          collate_fn=self._val_test_collate_fn, 
                          shuffle=False, 
                          drop_last=True, 
                          pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=int(self.batch_size/4), 
                          num_workers=self.num_workers, 
                          collate_fn=self._val_test_collate_fn, 
                          shuffle=False, 
                          drop_last=True, 
                          pin_memory=True)

    def predict_dataloader(self):
        return DataLoader(self.pred_dataset, 
                          batch_size=int(self.batch_size), 
                          num_workers=self.num_workers, 
                          collate_fn=self._pred_collate_fn, 
                          shuffle=False, 
                          drop_last=True, 
                          pin_memory=True)