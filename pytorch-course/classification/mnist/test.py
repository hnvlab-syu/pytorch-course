import argparse

import torch
from torch import nn
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

from src.dataset import get_mnist
from src.model import NeuralNetwork


parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu", help="device for test")
args = parser.parse_args()


def predict(test_data: Dataset, model: nn.Module, device: str) -> None:
    """
    Prediction the example image from MNIST dataset using a trained neural network

    Args:
        test_data (Dataset): dataset for test
        model (nn.Module): model for test
        device (str): device for test

    Returns:
        None
    """

    model.eval()
    
    image = test_data[0][0]
    plt.imshow(image.permute(1, 2, 0))
    plt.savefig("output_image.png")
    image = image.to(device)
    image = image.unsqueeze(0)
    target = test_data[0][1]
    target = torch.tensor(target)
    
    with torch.no_grad():
        pred = model(image)
        predicted = pred[0].argmax(0)
        actual = target
        print(f'Predicted: "{predicted}", Acutal: "{actual}"')
        
        
def test(device: str) -> None:
    """
    Pytorch test pipeline

    Args:
        device (str): device for test.

    Returns:
        None
    """
    num_classes = 10

    data_dir = 'data'
    _, test_data = get_mnist(data_dir)

    model = NeuralNetwork(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load('mnist-net.pth'))

    predict(test_data, model, device)

if __name__ == "__main__":
    test(device=args.device)