import argparse

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms

from src.dataset import ImagenetDataset
from src.model import VGG16model


parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu", help="학습에 사용되는 장치")
args = parser.parse_args()


def test(device):
    image_dir = '../dataset'

    num_classes = 10

    test_data = ImagenetDataset(image_dir)

    model = VGG16model(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load('fashion-mnist-net.pth'))

    model.eval()
    image = test_data[0][0].to(device)
    image = image.unsqueeze(0)
    target = test_data[0][1].to(device)
    with torch.no_grad():
        pred = model(image)
        predicted = pred[0].argmax(0)
        actual = target
        print(f'Predicted: "{predicted}", Actual: "{actual}"')


if __name__ == '__main__':
    test(args.device)
