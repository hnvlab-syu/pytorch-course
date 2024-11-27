import os
import torch
from PIL import Image
from torch.utils.data import DataLoader, random_split
import lightning as L
import numpy as np
from .components.cocoposedataset import CocoPoseDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


SEED = 36
L.seed_everything(SEED)

class PoseEstimationDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = '../data/', batch_size: int = 32, num_workers: int = 0, pin_memory: bool = False, mode: str = 'train'):
        super().__init__()
        self.mode = mode
        self.data_path = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        if self.mode == 'train':
            self.transform = A.Compose(
                [   
                    A.Resize(256, 256),  
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    ),
                    ToTensorV2()
                ],
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
                keypoint_params=A.KeypointParams(format='xy')
            )
        else:
            self.transform = A.Compose([
                A.Resize(256, 256),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])

    def setup(self, stage: str):
        if self.trainer is not None:
            if self.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.batch_size // self.trainer.world_size

        if self.mode == 'train':
            img_dir = os.path.join(self.data_path, 'val2017/val2017/')
            ann_file = os.path.join(self.data_path, 'person_keypoints_val2017.json')
            full_dataset = CocoPoseDataset(img_dir, ann_file, transform=self.transform)
            total_size = len(full_dataset)
            train_size = int(0.8 * total_size)
            val_size = int(0.1 * total_size)
            test_size = total_size - train_size - val_size
              
            train_data, val_data, test_data = random_split(
                full_dataset, 
                [train_size, val_size, test_size]
            )
        else:
            self.pred_dataset = [(np.array(Image.open(self.data_path).convert('RGB')), self.data_path)]

        if stage == 'fit':
            self.train_dataset = train_data
            self.val_dataset = val_data

        if stage == 'test':
            self.test_dataset = test_data

        if stage == 'predict':
            self.pred_dataset = self.pred_dataset

    def _train_collate_fn(self, batch):
        images, targets, filenames = zip(*batch) 
        images = torch.stack([img if isinstance(img, torch.Tensor) else self.transform(img) for img in images])
        return images, targets, filenames


    
    def _predict_collate_fn(self, batch):
        transformed_images = []
        filenames = []
        
        for img, filename in batch:
            transformed = self.transform(image=img)
            transformed_images.append(transformed['image'])
            filenames.append(filename)
        
        images = torch.stack(transformed_images)
        return images, None, filenames

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size_per_device, shuffle=True, collate_fn=self._train_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size_per_device, shuffle=False, collate_fn=self._train_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size_per_device, shuffle=False, collate_fn=self._train_collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.pred_dataset, batch_size=1, shuffle=False, collate_fn=self._predict_collate_fn)




if __name__ == "__main__":
    _ = PoseEstimationDataModule()