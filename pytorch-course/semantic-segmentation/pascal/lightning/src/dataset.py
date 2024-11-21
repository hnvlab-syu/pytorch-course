import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import lightning as L

SEED = 36
L.seed_everything(SEED)


def preprocess_mask(mask):
    mask = np.array(mask)
    mask[mask == 255] = 0   
    return torch.tensor(mask, dtype=torch.long)


class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, image_list, transform=None, mask_transform=None):
        self.root_dir = root_dir
        self.image_list = image_list
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_dir = os.path.join(root_dir, 'JPEGImages')
        self.mask_dir = os.path.join(root_dir, 'SegmentationClass')

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, f'{image_name}.jpg')
        
        image = Image.open(img_path).convert("RGB")   

        mask_path = os.path.join(self.mask_dir, f'{image_name}.png')
        mask = Image.open(mask_path)

        
        if self.transform and not isinstance(image, torch.Tensor):
            image = self.transform(image)
        if self.mask_transform and not isinstance(mask, torch.Tensor):
            mask = self.mask_transform(mask)

        

        mask = preprocess_mask(mask)
        return image, mask

class VOCDataModule(L.LightningDataModule):
    def __init__(self, data_path: str = './data/VOC2012', batch_size: int = 32, mode: str = 'train'):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.mode = mode  
        self.image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.NEAREST),
        ])

    def setup(self, stage: str = None):
        if stage == 'fit' or stage is None:
            train_set_path = os.path.join(self.data_path, 'ImageSets', 'Segmentation', 'train.txt')
            val_set_path = os.path.join(self.data_path, 'ImageSets', 'Segmentation', 'val.txt')
            with open(train_set_path, 'r') as f:
                train_list = f.read().splitlines()
            with open(val_set_path, 'r') as f:
                val_list = f.read().splitlines()

            self.train_dataset = VOCDataset(
                root_dir=self.data_path,
                image_list=train_list,
                transform=self.image_transform,
                mask_transform=self.mask_transform
            )
            self.val_dataset = VOCDataset(
                root_dir=self.data_path,
                image_list=val_list,
                transform=self.image_transform,
                mask_transform=self.mask_transform
            )

        if stage == 'test' or stage is None:
            test_set_path = os.path.join(self.data_path, 'ImageSets', 'Segmentation', 'trainval.txt')
            with open(test_set_path, 'r') as f:
                test_list = f.read().splitlines()

            self.test_dataset = VOCDataset(
                root_dir=self.data_path,
                image_list=test_list,
                transform=self.image_transform,
                mask_transform=self.mask_transform
            )

        if stage == 'predict':
           
            self.pred_dataset = [self.data_path]  
    def _train_collate_fn(self, batch):
        images, masks = zip(*batch)
        images = torch.stack(images)  
        masks = torch.stack(masks)    
        return images, masks

    def _predict_collate_fn(self, batch):
        img_path = batch[0]
        img = Image.open(img_path).convert("RGB")
        input_tensor = self.image_transform(img)
        return input_tensor.unsqueeze(0)  

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self._train_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self._train_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self._train_collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.pred_dataset, batch_size=1, collate_fn=self._predict_collate_fn)
