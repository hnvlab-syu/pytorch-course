import os
from typing import Callable, Optional, Sequence

import numpy as np
from PIL import Image
import pandas as pd

from torch import Tensor
from torch.utils.data import Dataset


class PascalVOC2012Dataset(Dataset):
    def __init__(
            self,
            image_dir: os.PathLike,
            label_dir: os.PathLike,
            csv_path: os.PathLike,
            num_classes: int,
            transform: Optional[Sequence[Callable]] = None,
            mask_transform: Optional[Sequence[Callable]] = None,
    ) -> None:
        super().__init__()

        df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.num_classes = num_classes

        self.image_ids = df['image_id'].tolist()

        self.transform = transform
        self.mask_transform = mask_transform

    # __len__ 초기화
    def __len__(self) -> int:
        return len(self.image_ids)
    
    # __getitem__ 초기화
    def __getitem__(self, index: int) -> Tensor:
        image_id = self.image_ids[index]

        # image, mask 가져오기
        image = Image.open(os.path.join(self.image_dir, f'{image_id}.jpg')).convert("RGB")
        mask = Image.open(os.path.join(self.label_dir, f'{image_id}.png'))

        # mask ndarray로 변경
        mask = np.asarray(mask)
        height, width = mask.shape

        # class_id에 맞는 구역 mask
        target = np.zeros((height, width, self.num_classes+1))
        for class_id in range(self.num_classes):
            target[mask == class_id+1, class_id+1] = 1
        
        # image, target transform
        if self.transform is not None:
            image = self.transform(image)
        if self.mask_transform is not None:
            target = self.mask_transform(target)

        # meta_data 설정
        meta_data = {
            'image_id': image_id,
            'height': height,
            'width': width
        }

        return image, target, meta_data