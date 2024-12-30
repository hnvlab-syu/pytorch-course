import os
import glob
import numpy as np
import torch
import lightning as L

from PIL import Image
from typing import Optional
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from basicsr.data.transforms import augment, paired_random_crop


SEED = 36
L.seed_everything(SEED)

class DIV2KDataModule(L.LightningDataModule):
    def __init__(
        self,
        lr_dir: str,
        hr_dir: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        mode: str,
        upscale: int,
    ):
        super().__init__()
        self.mode = mode
        if self.mode == 'train':    # train(val, test) or prediction
            self.lr_dataset = sorted(glob.glob(os.path.join(lr_dir, "*.png")))   # */*.png   # list format
            self.hr_dataset = sorted(glob.glob(os.path.join(hr_dir, "*.png")))

            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])

            if not self.lr_dataset or not self.hr_dataset:
                raise RuntimeError(f"No images found in {lr_dir} or {hr_dir}")
            if len(self.lr_dataset) != len(self.hr_dataset):
                raise RuntimeError("Number of LR and HR images don't match")
        else:
            self.data_path = lr_dir     # example.jpg 경로
            batch_size = 1

            self.transform = transforms.Compose([
                # transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.upscale = upscale

        self.gt_size = 160
        self.use_hflip = True
        self.use_rot = True 

    def setup(self, stage: Optional[str] = None):
        if self.mode == 'train':
            # x: lr, y: hr
            train_x, val_x, train_y, val_y = train_test_split(self.lr_dataset, self.hr_dataset, test_size=0.2, random_state=SEED)
            val_x, test_x, val_y, test_y = train_test_split(val_x, val_y, test_size=0.5, random_state=SEED)

            train_data = [(x, y) for x, y in zip(train_x, train_y)]
            val_data = [(x, y) for x, y in zip(val_x, val_y)]
            test_data = [(x, y) for x, y in zip(test_x, test_y)]
        else:
            pred_data = [Image.open(self.data_path).convert('RGB')]

        # trainer
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
            self.pred_dataset = pred_data

    def _train_collate_fn(self, batch):
        lr_list = []
        hr_list = []
        scale = self.upscale

        for lr_img, hr_img in batch:
            lr_img= Image.open(lr_img).convert('RGB')
            hr_img = Image.open(hr_img).convert('RGB')

            # PIL Image-> numpy array: BasicSR의 paired_random_crop(), augment()의 입력 형식을 맞추기 위함
            lr_np = np.array(lr_img) / 255.0  # [0, 1]로 정규화
            hr_np = np.array(hr_img) / 255.0

            # Augmentation: random crop
            gt_size = self.gt_size
            hr_np, lr_np = paired_random_crop(hr_np, lr_np, gt_size, scale)

            # Augmentation: flip, rotation
            hr_np, lr_np = augment([hr_np, lr_np], hflip=True, rotation=True)

            # numpy array-> PIL Image
            lr_img = Image.fromarray((lr_np * 255).astype(np.uint8))
            hr_img = Image.fromarray((hr_np * 255).astype(np.uint8))

            # transform (totensor)
            if self.transform:
                lr_img = self.transform(lr_img)
                hr_img = self.transform(hr_img)

            lr_list.append(lr_img)
            hr_list.append(hr_img)

        return torch.stack(lr_list), torch.stack(hr_list)
    
    # def _val_test_collate_fn(self, batch):
    #     img = batch[0]
    #     input = self.transform(img).unsqueeze(0)
    #     return input    # , np.array(img)     # 변환된 이미지, 원본 이미지

    def _predict_collate_fn(self, batch):
        img = batch[0]
        input = self.transform(img).unsqueeze(0)
        return input
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size,  
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory, 
            collate_fn=self._train_collate_fn,
            shuffle=True)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory, 
            collate_fn=self._train_collate_fn)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory,
            collate_fn=self._train_collate_fn)

    def predict_dataloader(self):
        return DataLoader(
            self.pred_dataset, 
            batch_size=self.batch_size,   # 1 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory,
            collate_fn=self._predict_collate_fn)
