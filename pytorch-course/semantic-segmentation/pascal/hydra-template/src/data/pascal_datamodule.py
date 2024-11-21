from typing import Any, Optional
import os
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from lightning import LightningDataModule, seed_everything
import numpy as np

SEED = 36
seed_everything(SEED)


def preprocess_mask(mask: Image.Image) -> torch.Tensor:
    mask = np.array(mask)
    mask[mask == 255] = 0
    return torch.tensor(mask, dtype=torch.long)


class VOCDataset(Dataset):
    def __init__(self, root_dir: str, image_list: list, transform=None, mask_transform=None, is_predict: bool = False):
        self.root_dir = root_dir
        self.image_list = image_list
        self.transform = transform
        self.mask_transform = mask_transform
        self.is_predict = is_predict
        self.image_dir = os.path.join(root_dir, 'JPEGImages') if not is_predict else root_dir

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, idx: int) -> tuple:
        image_name = self.image_list[idx]
        
        if self.is_predict:
            
            image = Image.open(self.root_dir).convert("RGB")
        else:
            
            img_path = os.path.join(self.image_dir, f"{image_name}.jpg")
            image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.is_predict:
            return image

        mask = Image.open(os.path.join(self.root_dir, "SegmentationClass", f"{image_name}.png"))
        if self.mask_transform:
            mask = self.mask_transform(mask)
        mask = preprocess_mask(mask)
        
        return image, mask


class VOCDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data/VOC2012",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        print(f"Data directory: {self.hparams.data_dir}")
        self.train_list, self.val_list, self.test_list = [], [], []
        self.image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.NEAREST),
        ])

    def setup(self, stage: Optional[str] = None) -> None:
        train_file = os.path.join(self.hparams.data_dir, 'ImageSets', 'Segmentation', 'train.txt')
        val_file = os.path.join(self.hparams.data_dir, 'ImageSets', 'Segmentation', 'val.txt')
        test_file = os.path.join(self.hparams.data_dir, 'ImageSets', 'Segmentation', 'trainval.txt')



        if stage in (None, "fit"):
            with open(train_file, "r") as f:
                self.train_list = f.read().splitlines()
            with open(val_file, "r") as f:
                self.val_list = f.read().splitlines()
            self.train_dataset = VOCDataset(
                root_dir=self.hparams.data_dir,
                image_list=self.train_list,
                transform=self.image_transform,
                mask_transform=self.mask_transform,
            )
            self.val_dataset = VOCDataset(
                root_dir=self.hparams.data_dir,
                image_list=self.val_list,
                transform=self.image_transform,
                mask_transform=self.mask_transform,
            )

        if stage in (None, "test"):
            with open(test_file, "r") as f:
                self.test_list = f.read().splitlines()
            self.test_dataset = VOCDataset(
                root_dir=self.hparams.data_dir,
                image_list=self.test_list,
                transform=self.image_transform,
                mask_transform=self.mask_transform,
            )

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        predict_dataset = VOCDataset(
            root_dir=self.hparams.data_dir,
            image_list=["dummy"],  
            transform=self.image_transform,
            mask_transform=None,
            is_predict=True
        )
        return DataLoader(
            dataset=predict_dataset,
            batch_size=1, 
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
    


if __name__ == "__main__":
    dm = VOCDataModule(data_dir="./data/VOC2012", batch_size=16)
    dm.setup("fit")
    print(f"Train samples: {len(dm.train_list)}, Val samples: {len(dm.val_list)}")
