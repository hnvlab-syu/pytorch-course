import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms

from src.dataset import Cifar10Dataset
from src.model import LeNet
from src.utils import CLASSES, split_dataset


def predict(test_data: Dataset, model: nn.Module) -> None:
    """학습한 뉴럴 네트워크로 CIFAR-10 데이터셋을 분류합니다.

    :param test_data: 추론에 사용되는 데이터셋
    :type test_data: Dataset
    :param model: 추론에 사용되는 모델
    :type model: nn.Module
    """
    model.eval()
    image = test_data[0][0].unsqueeze(0)
    target = test_data[0][1]
    with torch.no_grad():
        pred = model(image)
        predicted = CLASSES[pred[0].argmax(0)]
        actual = CLASSES[target]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')


def test():
    image_dir = 'data/train'
    test_csv_path = 'data/test_answer.csv'

    num_classes = 10

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_data = Cifar10Dataset(
        image_dir,
        test_csv_path,
        transform=transform
    )

    model = LeNet(num_classes=num_classes)
    model.load_state_dict(torch.load('cifar-net-lenet.pth'))

    predict(test_data, model)


if __name__ == '__main__':
    test()
