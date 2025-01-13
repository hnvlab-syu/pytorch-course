# COCO train, val, test 데이터 각각 사용한 버전
import os
import json
import glob
import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import albumentations as A

from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader


class COCODataModule(pl.LightningDataModule):
    def __init__(self, 
        datasets_path, 
        annots_path, 
        batch_size,
        num_workers,
        mode):
        super().__init__()
        self.datasets_path = datasets_path
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
        # test/predict용 transform (bbox params 제외)
        # self.test_transforms = A.Compose([])

    # def prepare_data(self):
    #     # download

    def setup(self, stage: str):
        train_csv_path = os.path.join(self.annots_path, 'train_annotations.csv')
        val_csv_path = os.path.join(self.annots_path, 'val_annotations.csv')
    
        if not os.path.exists(train_csv_path) or not os.path.exists(val_csv_path):
            print("Creating CSV files from COCO annotations...")
            # COCO JSON 로드
            with open(os.path.join(self.annots_path, "instances_train2017.json"), 'r') as f:
                self.train_annots = json.load(f)
            with open(os.path.join(self.annots_path, "instances_val2017.json"), 'r') as f:
                self.val_annots = json.load(f)
    
            train_data = []
            for img in tqdm(self.train_annots['images']):
                img_id = img['id']
                img_path = os.path.join(self.datasets_path, "train2017", img['file_name'])
                annots = [x for x in self.train_annots['annotations'] if x['image_id'] == img_id]
                
                if annots:  # annotation이 있는 이미지만 저장
                    train_data.append({
                        'image_path': img_path,
                        'boxes': str([x['bbox'] for x in annots]),  # list를 문자열로 변환
                        'labels': str([x['category_id'] for x in annots])
                    })
            
            val_data = []
            for img in tqdm(self.val_annots['images']):
                img_id = img['id']
                img_path = os.path.join(self.datasets_path, "val2017", img['file_name'])
                annots = [x for x in self.val_annots['annotations'] if x['image_id'] == img_id]
                
                if annots:
                    val_data.append({
                        'image_path': img_path,
                        'boxes': str([x['bbox'] for x in annots]),
                        'labels': str([x['category_id'] for x in annots])
                    })
    
            pd.DataFrame(train_data).to_csv(train_csv_path, index=False)
            pd.DataFrame(val_data).to_csv(val_csv_path, index=False)
            print("CSV files created successfully")


        if stage == "fit":
            print("Loading annotations from CSV...")
            train_df = pd.read_csv(train_csv_path)
            val_df = pd.read_csv(val_csv_path)
    
            # DataFrame을 원하는 형식으로 변환
            self.train_data = []
            for _, row in train_df.iterrows():
                boxes = eval(row['boxes'])  # 문자열을 list로 변환
                labels = eval(row['labels'])
                self.train_data.append({
                    'image_path': row['image_path'],
                    'boxes': boxes,
                    'labels': labels
                })
    
            self.val_data = []
            for _, row in val_df.iterrows():
                boxes = eval(row['boxes'])
                labels = eval(row['labels'])
                self.val_data.append({
                    'image_path': row['image_path'],
                    'boxes': boxes,
                    'labels': labels
                })
            print('---------------------')
            print(f"Train size: {len(self.train_data)}, Val size: {len(self.val_data)}")

        if stage == "test": # stage==mode??????????????????????????????????
            self.test_data = sorted(glob.glob(os.path.join(self.datasets_path, "test2017", "*.{jpg,jpeg,png}")))
            print('---------------------')
            print(f"Test size: {len(self.test_data)}")

        if stage == "predict":
            self.pred_data = sorted(glob.glob(os.path.join(self.datasets_path, "unlabeled2017", "*.{jpg,jpeg,png}")))
            print('---------------------')
            print(f"Predict size: {len(self.pred_data)}")    


    def _collate_fn(self, batch):
        images = []
        targets = []
        
        for b in batch:  # def setup처리 후의 data
            try:
                if isinstance(b, str):  # test/predict (image path만 주어진 경우)
                    img = Image.open(b).convert('RGB')
                    img = np.array(img)
                    
                    if self.trainer.testing:
                        transformed = self.val_transforms(image=img)
                    else:
                        transformed = self.val_transforms(image=img)
                    
                    img = transformed['image']
                    img = img.transpose(2, 0, 1)
                    img = img / 255.0
                    img = torch.Tensor(img)
                    
                    images.append(img)
                    targets.append({})  # 타겟 공란
                    continue
    
                elif isinstance(b, dict):    # train/val ????????????list아니고?
                    img = Image.open(b['image_path']).convert('RGB')
                    img = np.array(img)
                    
                    boxes = np.array(b['boxes'], dtype=np.float32)
                    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
                    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
                    
                    labels = np.array(b['labels'], dtype=np.int64)
                
                    if self.trainer.training:   # Lightning의 training 상태
                        transformed = self.train_transforms(image=img, bboxes=boxes, class_labels=labels)
                    else:
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

                else:
                    raise ValueError(f"Unexpected data format in batch: {b}")

            except Exception as e:  # ValueError: y_max is less than or equal to y_min for bbox [ 0.4635156  0.8090208  0.465125   0.8090208 58.       ].
                print(f"Error processing batch element: {b}\n{e}")
                continue
    
        # print('---------------images:', images)
        # print('---------------targets: ', targets)
        return images, targets


    def train_dataloader(self):
        return DataLoader(self.train_data, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          collate_fn=self._collate_fn, 
                          shuffle=True, 
                          drop_last=True, 
                          pin_memory=True)
        
    def val_dataloader(self):
        return DataLoader(self.val_data, 
                          batch_size=int(self.batch_size/4), 
                          num_workers=self.num_workers, 
                          collate_fn=self._collate_fn, 
                          shuffle=False, 
                          drop_last=True, 
                          pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, 
                          batch_size=int(self.batch_size/4), 
                          num_workers=self.num_workers, 
                          collate_fn=self._collate_fn, 
                          shuffle=False, 
                          drop_last=True, 
                          pin_memory=True)

    def predict_dataloader(self):
        return DataLoader(self.pred_data, 
                          batch_size=int(self.batch_size), 
                          num_workers=self.num_workers, 
                          collate_fn=self._collate_fn, 
                          shuffle=False, 
                          drop_last=True, 
                          pin_memory=True)
        

   




    def setup(self, stage: str):
        with open(os.path.join(self.annots_path, "instances_val2017.json"), 'r') as f:
            self.val_annots = json.load(f)

        val_data = []
        for img in tqdm(self.val_annots['images']):
            img_id = img['id']
            img_path = os.path.join(self.data_path, "val2017", img['file_name'])
            annots = [x for x in self.val_annots['annotations'] if x['image_id'] == img_id]
            
            if annots:
                val_data.append({
                    'image_path': img_path,
                    'boxes': str([x['bbox'] for x in annots]),
                    'labels': str([x['category_id'] for x in annots])
                })

        if self.mode == 'train':
            # 데이터셋 분할: 60% train, 20% val, 20% test
            train_data, temp_data = train_test_split(val_data, test_size=0.4, random_state=SEED)
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
            # 테스트 데이터에서 예측할 이미지 경로만 수집
            self.pred_dataset = [
                os.path.join(self.data_path, "test2017", fname)
                for fname in os.listdir(os.path.join(self.data_path, "test2017"))
                if fname.endswith(('.jpg', '.jpeg', '.png'))
            ]
            print('---------------------')
            print(f"Predict size: {len(self.pred_dataset)}")




    def _collate_fn(self, batch):
        images = []
        targets = []
        
        for b in batch:  # def setup처리 후의 data
            try:
                if isinstance(b, str):  # test/predict (image path만 주어진 경우)
                    img = Image.open(b).convert('RGB')
                    img = np.array(img)
                    
                if self.val_transforms:
                    transformed = self.val_transforms(image=img) #????????
                    
                    img = transformed['image']
                    img = img.transpose(2, 0, 1)
                    img = img / 255.0
                    img = torch.Tensor(img)
                    
                    images.append(img)
                    targets.append({})  # 타겟 공란
                    continue
    
                elif isinstance(b, dict):    # train/val ????????????list아니고?
                    img = Image.open(b['image_path']).convert('RGB')
                    img = np.array(img)
                    
                    boxes = np.array(b['boxes'], dtype=np.float32)
                    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
                    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
                    
                    labels = np.array(b['labels'], dtype=np.int64)
                
                    if self.trainer.training:   # Lightning의 training 상태
                        transformed = self.train_transforms(image=img, bboxes=boxes, class_labels=labels)
                    else:
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

                else:
                    raise ValueError(f"Unexpected data format in batch: {b}")

            except Exception as e:  # ValueError: y_max is less than or equal to y_min for bbox [ 0.4635156  0.8090208  0.465125   0.8090208 58.       ].
                print(f"Error processing batch element: {b}\n{e}")
                continue
    
        # print('---------------images:', images)
        # print('---------------targets: ', targets)
        return images, targets