class COCODataModule(pl.LightningDataModule):
    def __init__(self, 
        datasets_path, 
        annt_path, 
        batch_size,
        num_workers):
        super().__init__()
        self.datasets_path = datasets_path
        self.annt_path = annt_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.train_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

        self.else_transforms = A.Compose([
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    def setup(self, stage: str):
        # annotation 파일 로드
        with open(self.annt_path, 'r') as f:
            self.annotations = json.load(f)
            
        if stage == "fit":
            # train 데이터 준비
            self.train_data = []
            for img in self.annotations['images']:
                if 'train2017' in img['file_name']:  # train 이미지만 필터링
                    img_id = img['id']
                    img_path = os.path.join(self.datasets_path, "train2017", img['file_name'])
                    
                    # 해당 이미지의 bbox와 class 찾기
                    annots = [x for x in self.annotations['annotations'] 
                             if x['image_id'] == img_id]
                    
                    self.train_data.append({
                        'image_path': img_path,
                        'boxes': [x['bbox'] for x in annots],
                        'labels': [x['category_id'] for x in annots]
                    })
            
            # val 데이터 준비
            self.val_data = []
            for img in self.annotations['images']:
                if 'val2017' in img['file_name']:  # validation 이미지만 필터링
                    img_id = img['id']
                    img_path = os.path.join(self.datasets_path, "val2017", img['file_name'])
                    
                    annots = [x for x in self.annotations['annotations'] 
                             if x['image_id'] == img_id]
                    
                    self.val_data.append({
                        'image_path': img_path,
                        'boxes': [x['bbox'] for x in annots],
                        'labels': [x['category_id'] for x in annots]
                    })

            print('---------------------')
            print(f"Train size: {len(self.train_data)}, Val size: {len(self.val_data)}")

        if stage == "predict":
            self.pred_data = sorted(glob.glob(os.path.join(self.datasets_path, "unlabeled2017", "*.{jpg,jpeg,png}")))

    def _train_val_collate_fn(self, batch):
        images = []
        targets = []
        
        for b in batch:
            # 이미지 로드 및 전처리
            img = Image.open(b['image_path']).convert('RGB')
            img = np.array(img)
            
            # bbox 포맷 변환 (COCO to PASCAL_VOC)
            boxes = np.array(b['boxes'], dtype=np.float32)
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
            
            labels = np.array(b['labels'], dtype=np.int64)
            
            # augmentation 적용
            if self.train_transforms:
                transformed = self.train_transforms(image=img, bboxes=boxes, class_labels=labels)
                img = transformed['image']
                boxes = transformed['bboxes']
                labels = transformed['class_labels']
            
            # 이미지 포맷 변환
            img = img.transpose(2, 0, 1)  # HWC to CHW
            img = img / 255.0  # normalize to [0, 1]
            img = torch.Tensor(img)
            
            images.append(img)
            targets.append({
                'boxes': torch.as_tensor(boxes, dtype=torch.float32),
                'labels': torch.as_tensor(labels, dtype=torch.int64)
            })
            
        return images, targets

    def train_dataloader(self):
        return DataLoader(
            self.train_data, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            collate_fn=self._train_val_collate_fn, 
            shuffle=True, 
            pin_memory=True
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_data, 
            batch_size=int(self.batch_size/4), 
            num_workers=self.num_workers, 
            collate_fn=self._train_val_collate_fn, 
            shuffle=False, 
            pin_memory=True
        )

    def predict_dataloader(self):
        return DataLoader(
            self.pred_data, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            collate_fn=self._predict_collate_fn, 
            shuffle=False, 
            pin_memory=True
        )
    




########################


class DataModule(pl.LightningDataModule):
    def setup(self, stage):
        # 1. annotation 파일 한 번만 로드
        with open(self.label_path, 'r') as f:
            self.annotations = json.load(f)
        
        # 2. 이미지-어노테이션 매핑 미리 생성
        self.train_data = []
        for img in self.annotations['images']:
            img_id = img['id']
            img_path = os.path.join(self.image_dir, img['file_name'])
            
            # 해당 이미지의 bbox와 class 찾기
            annots = [x for x in self.annotations['annotations'] 
                     if x['image_id'] == img_id]
            
            self.train_data.append({
                'image_path': img_path,
                'boxes': [x['bbox'] for x in annots],
                'labels': [x['category_id'] for x in annots]
            })

    def collate_fn(self, batch):
        # 3. collate_fn은 이미지 로드와 배치 처리만 담당
        images = []
        targets = []
        
        for b in batch:
            img = Image.open(b['image_path']).convert('RGB')
            if self.transform:
                img = self.transform(img)
            
            images.append(img)
            targets.append({
                'boxes': torch.tensor(b['boxes']),
                'labels': torch.tensor(b['labels'])
            })
            
        return images, targets
    





################################ coco preprocess -> csv로 저장해서 사용 전
import os
import json
import glob
import tqdm
import torch
import numpy as np
import pytorch_lightning as pl
import albumentations as A

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
        # print('----------------------')
        # print(f"Setup stage: {stage}")
        # print(f"Loading annotations from {self.annots_path}")
              
        # with open(os.path.join(self.annots_path, "instances_train2017.json"), 'r') as f:
        #     self.train_annots = json.load(f)
        # print("----------Loaded train annotations")

        # with open(os.path.join(self.annots_path, "instances_val2017.json"), 'r') as f:
        #     self.val_annots = json.load(f)
        # print("----------Loaded val annotations")
        print("Starting dataset setup...")
        try:
            print(f"Loading train annotations from {os.path.join(self.annots_path, 'instances_train2017.json')}")
            with open(os.path.join(self.annots_path, "instances_train2017.json"), 'r') as f:
                self.train_annots = json.load(f)
            print("Successfully loaded train annotations")
            print(f"Number of images in train: {len(self.train_annots['images'])}")
            print(f"Number of annotations in train: {len(self.train_annots['annotations'])}")
    
            print(f"Loading val annotations from {os.path.join(self.annots_path, 'instances_val2017.json')}")
            with open(os.path.join(self.annots_path, "instances_val2017.json"), 'r') as f:
                self.val_annots = json.load(f)
            print("Successfully loaded val annotations")
            print(f"Number of images in val: {len(self.val_annots['images'])}")
            print(f"Number of annotations in val: {len(self.val_annots['annotations'])}")
    
        except Exception as e:
            print(f"Error loading annotations: {str(e)}")
            raise
            
        if stage == "fit":
            print('!!')
            self.train_data = []
            for img in tqdm(self.train_annots['images']):
                
                img_id = img['id']
                img_path = os.path.join(self.datasets_path, "train2017", img['file_name'])  # (ex) file_name: 000000391895.jpg (확장자까지 포함)
                
                annots = [x for x in self.train_annots['annotations'] if x['image_id'] == img_id]
                
                self.train_data.append({
                    'image_path': img_path,
                    'boxes': [x['bbox'] for x in annots],
                    'labels': [x['category_id'] for x in annots]
                })
            
            self.val_data = []
            for img in self.val_annots['images']:
                img_id = img['id']
                img_path = os.path.join(self.datasets_path, "val2017", img['file_name'])
                
                annots = [x for x in self.val_annots['annotations'] if x['image_id'] == img_id]
                
                self.val_data.append({
                    'image_path': img_path,
                    'boxes': [x['bbox'] for x in annots],
                    'labels': [x['category_id'] for x in annots]
                })

            print('---------------------')
            print(f"Train size: {len(self.train_data)}, Val size: {len(self.val_data)}")

        if stage == "test":
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
        
        for b in batch:     # setup 처리 후의 train_data etc.
            # test/predict 시에는 image path만 있는 경우
            if isinstance(b, str):  # test/predict의 경우
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
                targets.append({})  # empty target
                continue

            # train/val의 경우
            img = Image.open(b['image_path']).convert('RGB')
            img = np.array(img)
            
            boxes = np.array(b['boxes'], dtype=np.float32)
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
            
            labels = np.array(b['labels'], dtype=np.int64)
        
            # Lightning의 training 상태로 transform 구분
            if self.trainer.training:   
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
        

   
    


# -----------------------------------------------------------
def _collate_fn(self, batch):
    images = []
    targets = []

    for b in batch:
        try:
            if isinstance(b, str):  # Test/Predict 데이터
                img = Image.open(b).convert('RGB')
                img = np.array(img)
                
                # Test or Predict 전용 Transform (val_transforms 사용)
                transformed = self.val_transforms(image=img)
                img = transformed['image']
                img = img.transpose(2, 0, 1) / 255.0
                img = torch.Tensor(img)
                
                images.append(img)
                targets.append({})  # Test/Predict에서는 빈 타겟 사용
            elif isinstance(b, dict) and 'image_path' in b and 'boxes' in b and 'labels' in b:  # Train/Val 데이터
                img = Image.open(b['image_path']).convert('RGB')
                img = np.array(img)
                
                # COCO 형식의 bbox를 [xmin, ymin, xmax, ymax]로 변환
                boxes = np.array(b['boxes'], dtype=np.float32)
                boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
                boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
                
                labels = np.array(b['labels'], dtype=np.int64)
                
                # Training 여부에 따라 Transform 적용
                if self.trainer.training:
                    transformed = self.train_transforms(image=img, bboxes=boxes, class_labels=labels)
                else:
                    transformed = self.val_transforms(image=img, bboxes=boxes, class_labels=labels)
                
                img = transformed['image']
                boxes = transformed['bboxes']
                labels = transformed['class_labels']
                
                img = img.transpose(2, 0, 1) / 255.0
                img = torch.Tensor(img)
                
                images.append(img)
                targets.append({
                    'boxes': torch.tensor(boxes, dtype=torch.float32),
                    'labels': torch.tensor(labels, dtype=torch.int64),
                })
            else:
                raise ValueError(f"Unexpected data format in batch: {b}")

        except Exception as e:
            print(f"Error processing batch element: {b}\n{e}")
            # 여기서 예외를 처리하거나 무시하고 계속 진행할 수 있습니다.
            continue

    return images, targets











# main.py coco evaluation tool로 test/predict 구현
if args.mode == 'evaluate':  # 'test'에서 'evaluate'로 변경
    model = model.load_from_checkpoint(args.checkpoint)
    trainer = pl.Trainer(gpus=1, precision=16)
    
    # ground truth가 있는 validation/test set으로 평가
    output_lists = trainer.predict(model, data_module)
    
    # COCO evaluation...
    coco_gt = COCO(args.valid_annt_path)
    coco_pred = coco_gt.loadRes('predict.json')
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    
elif args.mode == 'predict':  # 실제 예측을 위한 새로운 모드 추가
    model = model.load_from_checkpoint(args.checkpoint)
    trainer = pl.Trainer(gpus=1, precision=16)
    
    # annotation 없이 새로운 이미지에 대해서만 예측
    outputs = trainer.predict(model, data_module)
    # 결과 저장 로직...