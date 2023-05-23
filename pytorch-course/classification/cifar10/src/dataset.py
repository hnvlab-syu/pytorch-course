from collections import defaultdict
from glob import glob
import os

import numpy as np

from torch.utils.data import Dataset

from src.utils import unpickle


class Cifar10Dataset(Dataset):
    def __init__(self, data_dir, transform, mode='train'):
        super().__init__()

        self.data = defaultdict(list)
        data_paths = glob(os.path.join(data_dir, 'data_batch_*')) if mode == 'train' else [os.path.join(data_dir, 'test_batch')]
        for data_path in data_paths:
            batch = unpickle(data_path)
            self.data['data'].append(batch['data'])
            self.data['labels'].append(batch['labels'])

        self.data['data'] = np.concatenate(self.data['data'])

        self.transform = transform

    def __len__(self):
        return len(self.data['labels'])

    def __getitem__(self, index):
        image = self.data['data'][index].reshape(32, 32, 3)
        label = self.data['labels'][index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label
