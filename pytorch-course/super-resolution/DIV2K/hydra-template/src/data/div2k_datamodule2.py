import os
import glob
import numpy as np
import torch
import lightning as L

from PIL import Image
from typing import Optional, Tuple
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from basicsr.data.transforms import augment, paired_random_crop


SEED = 36
L.seed_everything(SEED)

class DIV2KDataset(Dataset):
    def __init__(
            self, 
            x: list,
            y: list, 
            transform,
            phase,
        ):  
        super().__init__()
        self.lr = x
        self.hr = y
        self.transform = transform
        self.phase = phase
        self.gt_size = 160
        self.use_hflip = True
        self.use_rot = True 

    def __len__(self) -> int:
        return len(self.lr)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = 4

        lr_image = Image.open(self.lr[idx]).convert('RGB')
        hr_image = Image.open(self.hr[idx]).convert('RGB')

        # PIL Image-> numpy array
        lr_np = np.array(lr_image) / 255.0  # [0, 1]로 정규화
        hr_np = np.array(hr_image) / 255.0

        # training phase에서만 augmentation
        if self.phase == 'train': 
            # random crop
            gt_size = self.gt_size 
            hr_np, lr_np = paired_random_crop(hr_np, lr_np, gt_size, scale, self.hr[idx])

            # flip, rotation
            hr_np, lr_np = augment([hr_np, lr_np], hflip=True, rotation=True)    

        # numpy array-> PIL Image
        lr_image = Image.fromarray((lr_np * 255).astype(np.uint8))
        hr_image = Image.fromarray((hr_np * 255).astype(np.uint8))

        # transform (totensor)
        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)

        return lr_image, hr_image


class DIV2KDataModule(L.LightningDataModule):
    def __init__(
        self,
        lr_dir: str,
        hr_dir: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        mode: str,
    ):
        super().__init__()
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mode = mode
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def prepare_data(self):
        if self.mode == 'train':    # train(val, test) or prediction
            self.lr_dataset = sorted(glob.glob(os.path.join(self.lr_dir, "*.png")))   # */*.png   # list format
            self.hr_dataset = sorted(glob.glob(os.path.join(self.hr_dir, "*.png")))

            if not self.lr_dataset or not self.hr_dataset:
                raise RuntimeError(f"No images found in {self.lr_dir} or {self.hr_dir}")

            if len(self.lr_dataset) != len(self.hr_dataset):
                raise RuntimeError("Number of LR and HR images don't match")
        else:
            self.data_path = self.lr_dir
            self.batch_size = 1

    def setup(self, stage: Optional[str] = None):
        if self.mode == 'train':
            # x: lr, y: hr
            train_x, val_x, train_y, val_y = train_test_split(self.lr_dataset, self.hr_dataset, test_size=0.2, random_state=SEED)
            val_x, test_x, val_y, test_y = train_test_split(val_x, val_y, test_size=0.5, random_state=SEED)
        else:
            pred_data = [Image.open(self.data_path).convert('RGB')]


        if stage == "fit":
            self.train_dataset = DIV2KDataset(train_x, train_y, self.transform, 'train')
            self.val_dataset = DIV2KDataset(val_x, val_y, self.transform, 'val')
            print('---------------------')
            print(f"Train size: {len(self.train_dataset)}, Val size: {len(self.val_dataset)}")

        if stage == "test":
            test_x, test_y = self.lr_dataset, self.hr_dataset
            self.test_dataset = DIV2KDataset(test_x, test_y, self.transform, 'test')
            print('---------------------')
            print(f"Test size: {len(self.test_dataset)}")

        if stage == "predict":
            self.pred_dataset = DIV2KDataset(pred_data, self.transform, 'predict') ###################################???

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=True, 
                          num_workers=self.num_workers, 
                          pin_memory=True, )
                          # collate_fn=self.collate_fn) # num_workers=4

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          pin_memory=True, )
                          # collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          pin_memory=True, )
                          # collate_fn=self.collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.pred_dataset, 
                          batch_size=self.batch_size,   # 1 
                          num_workers=self.num_workers, 
                          pin_memory=True, )
                          # collate_fn=self.collate_fn)

