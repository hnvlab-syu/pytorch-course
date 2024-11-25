import os
from typing import Callable, Optional, Sequence

from PIL import Image
import pandas as pd

from torch import Tensor


class KaggleSRDataset(Dataset):
    def __init__(
        self,
        lr_dir: os.PathLike,
        hr_dir: os.PathLike,
        csv_path: os.PathLike,
        transform: Optional[Sequence[Callable]] = None,
    ) -> None:
        super().__init__()
        
        df = pd.read_csv(csv_path)
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir

        self.image_ids = df['image_id'].tolist()

        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Tensor:
        image_id = self.image_ids[index]

        lr_image = Image.open(os.path.join(self.lr_dir, f'{image_id}.png')).convert('RGB')
        hr_image = Image.open(os.path.join(self.hr_dir, f'{image_id}.png')).convert('RGB')

        if self.transform is not None:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)

        return lr_image, hr_image, image_id