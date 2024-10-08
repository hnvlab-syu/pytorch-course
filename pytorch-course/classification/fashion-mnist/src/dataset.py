from torchvision import datasets
from torchvision.transforms import ToTensor


def get_fashion_mnist(dir:str='data'):
    training_data = datasets.FashionMNIST(
        root=dir,
        train=True,
        download=True,
        transform=ToTensor()
    )
    
    test_data = datasets.FashionMNIST(
        root=dir,
        train=False,
        download=True,
        transform=ToTensor()
    )
    
    return training_data, test_data