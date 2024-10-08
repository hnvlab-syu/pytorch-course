from torchvision import datasets
from torchvision.transforms import ToTensor
from typing import Tuple


def get_mnist(dir: str = 'data') -> Tuple[datasets.VisionDataset, datasets.VisionDataset]:
    """
    get mnist dataset

    Args:
        dir (str): dataset download directory.

    Returns:
        tuple[datasets.VisionDataset, datasets.VisionDataset]:
            - training_data (datasets.VisionDataset): The MNIST training dataset.
            - test_data (datasets.VisionDataset): The MNIST test dataset
    """   
    training_data = datasets.MNIST(
        root=dir,
        train=True,
        download=True,
        transform=ToTensor()
    )
    test_data = datasets.MNIST(
        root=dir,
        train=False,
        download=True,
        transform=ToTensor()
    )

    return training_data, test_data