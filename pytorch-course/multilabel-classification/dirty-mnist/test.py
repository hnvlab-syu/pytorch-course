import torch
from torch import nn
from torch.utils.data import Dataset

from src.dataset import DirtyMnistDataset
from src.model import MultiLabelResNet
from src.utils import get_val_transform


def predict(test_data: Dataset, model: nn.Module) -> None:
    """학습한 뉴럴 네트워크로 Dirty-MNIST 데이터셋을 분류합니다.

    :param test_data: 추론에 사용되는 데이터셋
    :type test_data: Dataset
    :param model: 추론에 사용되는 모델
    :type model: nn.Module
    """
    from string import ascii_lowercase
    classes = list(ascii_lowercase)

    model.eval()
    image = test_data[1][0].unsqueeze(0)
    target = test_data[1][1]
    with torch.no_grad():
        pred = torch.sigmoid(model(image)) >= 0.5
        pred = pred.squeeze(0).nonzero()
        print(pred)
        print(target.nonzero()[0])
        predicted = [classes[p] for p in pred]
        actual = [classes[a] for a in target.nonzero()[0]]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')


def test():
    image_dir = 
    test_csv_path = 

    num_classes = 26

    test_data = DirtyMnistDataset(
        image_dir=image_dir,
        csv_path=test_csv_path,
        transform=get_val_transform()
    )

    model = MultiLabelResNet(num_classes=num_classes)
    model.load_state_dict(torch.load('dirty-mnist-resnet.pth'))

    predict(test_data, model)
