import argparse

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms

from src.dataset import FashionMnistDataset
from src.model import NeuralNetwork


parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu", help="학습에 사용되는 장치")
args = parser.parse_args()


def predict(test_data: Dataset, model: nn.Module, device) -> None:
    """학습한 뉴럴 네트워크로 FashionMNIST 데이터셋을 분류합니다.

    :param test_data: 추론에 사용되는 데이터셋
    :type test_data: Dataset
    :param model: 추론에 사용되는 모델
    :type model: nn.Module
    """
    classes = [
        'T-shirt/top',
        'Trouser',
        'Pullover',
        'Dress',
        'Coat',
        'Sandal',
        'Shirt',
        'Sneaker',
        'Bag',
        'Ankle boot',
    ]

    model.eval()
    image = test_data[0][0].to(device)
    image = image.unsqueeze(0)
    target = test_data[0][1].to(device)
    with torch.no_grad():
        pred = model(image)
        predicted = classes[pred[0].argmax(0)]
        actual = classes[target]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')


def test(device):
    image_dir = 'data/fashion-mnist/images'
    test_csv_path = 'data/fashion-mnist/test_answer.csv'

    num_classes = 10

    test_data = FashionMnistDataset(
        image_dir,
        test_csv_path
    )

    model = NeuralNetwork(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load('fashion-mnist-net.pth'))

    predict(test_data, model, device)


if __name__ == '__main__':
    test(args.device)
