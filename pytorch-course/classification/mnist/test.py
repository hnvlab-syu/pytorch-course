import torch
from torch import nn
from torch.utils.data import Dataset

from src.model import NeuralNetwork
from src.dataset import MnistDataset


def predict(test_data: Dataset, model: nn.Module, device: str) -> None:
    """학습한 뉴럴 네트워크로 MNIST 데이터셋을 분류합니다.

    :param test_data: 추론에 사용되는 데이터셋
    :type test_data: Dataset
    :param model: 추론에 사용되는 모델
    :type model: nn.Module
    """
    model.eval()
    image = test_data[0][0].to(device)
    image = image.unsqueeze(0)
    print(image.size())
    target = test_data[0][1].to(device)
    with torch.no_grad():
        pred = model(image)
        predicted = pred[0].argmax(0)
        actual = target
        print(f'Predicted: "{predicted}", Actual: "{actual}"')


def test():
    num_classes = 10

    test_data = MnistDataset("./data/MNIST - JPG - testing", transform=None)
    # print(test_data[0][0].shape)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = NeuralNetwork(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load('mnist-net.pth'))

    predict(test_data, model, device)


if __name__ == "__main__":
    test()