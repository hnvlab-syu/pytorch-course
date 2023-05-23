import os

import numpy as np
from PIL import Image
import pandas as pd

import torch
from torch.utils.data import Dataset


class FashionMnistDataset(Dataset):
    def __init__(self, image_dir, label_path):
        super().__init__()

        self.image_dir = image_dir
        self.labels = pd.read_csv(label_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image_id = self.labels.loc[index]
        image = Image.open(os.path.join(self.image_dir, f"{image_id['id']}.jpg"))
        label = image_id['label']

        image = np.array(image)

        image = torch.FloatTensor(image)
        label = torch.LongTensor([int(label)])

        return image, label
