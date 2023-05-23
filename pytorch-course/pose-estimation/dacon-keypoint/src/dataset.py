import os
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import pandas as pd

import torch
from torch import Tensor
from torch.utils.data import Dataset


class DaconKeypointDataset(Dataset):
    def __init__(
        self,
        image_dir: os.PathLike,
        csv_path: os.PathLike,
        transform: Optional[Sequence[Callable]] = None,
    ) -> None:
        super().__init__()
        self.image_dir = image_dir
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self) -> int:
        return self.df.shape[0]
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Dict]:
        image_id = self.df.iloc[index, 0]
        labels = np.array([1])
        keypoints = self.df.iloc[index, 1:].values.reshape(-1, 2).astype(np.int64)

        x1, y1 = min(keypoints[:, 0]), min(keypoints[:, 1])
        x2, y2 = max(keypoints[:, 0]), max(keypoints[:, 1])
        boxes = np.array([[x1, y1, x2, y2]], dtype=np.int64)

        image = Image.open(os.path.join(self.image_dir, image_id)).convert('RGB')
        image = np.asarray(image)

        targets ={
            'image': image,
            'bboxes': boxes,
            'labels': labels,
            'keypoints': keypoints
        }

        if self.transform is not None:
            targets = self.transform(**targets)

            image = targets['image']
            image = image / 255.0

            targets = {
                'boxes': torch.as_tensor(targets['bboxes'], dtype=torch.float32),
                'labels': torch.as_tensor(targets['labels'], dtype=torch.int64),
                'keypoints': torch.as_tensor(
                    np.concatenate([targets['keypoints'], np.ones((24, 1))], axis=1)[np.newaxis], dtype=torch.float32
                )
            }

        return image, targets, image_id


def collate_fn(batch: torch.Tensor) -> Tuple:
    return tuple(zip(*batch))
