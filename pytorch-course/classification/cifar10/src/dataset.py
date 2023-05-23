import os

from PIL import Image
import pandas as pd

from torch.utils.data import Dataset


class Cifar10Dataset(Dataset):
    def __init__(self, image_dir, label_path, transform):
        super().__init__()

        self.image_dir = image_dir
        self.labels = pd.read_csv(label_path)

        self.classes = [
            'plane',
            'car',
            'bird',
            'cat',
            'deer',
            'dog',
            'frog',
            'horse',
            'ship',
            'truck',
        ]

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image_id = self.labels.loc[index]
        image = Image.open(os.path.join(self.image_dir, f"{image_id['id']}.png")).convert('RGB')
        label = self.classes[image_id['label']]

        if self.transform is not None:
            image = self.transform(image)

        return image, label
