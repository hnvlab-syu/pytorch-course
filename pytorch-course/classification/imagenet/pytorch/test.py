import argparse

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.dataset import ImagenetDataset, get_transform
from src.model import create_model
from src.utils import set_seed, split_dataset


def test(args):
    device = args.device
    model_name = args.model_name
    image_size = args.image_size
    batch_size = args.batch_size

    data_dir = '../dataset'
    _, _, _, _, test_x, test_y = split_dataset(data_dir=data_dir)

    test_data = ImagenetDataset(
        image_dir=test_x,
        class_name=test_y,
        transform=get_transform(state='test', image_size=image_size)
    )

    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=0)

    model = create_model(model=model_name).to(device)
    model.load_state_dict(torch.load(f'best_epoch-imagenet-{model_name}.pth'))

    loss_fn = nn.CrossEntropyLoss()

    # validation
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    model.eval()
    total_test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch, (images, targets) in enumerate(test_dataloader):
            images = images.to(device)
            targets = targets.to(device)
            targets = torch.flatten(targets)

            preds = model(images)
            test_loss = loss_fn(preds, targets)

            total_test_loss += test_loss.item()
            correct += (preds.argmax(1) == targets).float().sum().item()

            if batch % 20 == 0:
                loss = test_loss.item()
                current = batch * len(images)
                print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')

    total_test_loss /= num_batches
    correct /= size
    print(f'Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {total_test_loss:>8f} \n')


if __name__ == '__main__':
    set_seed(36)

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", help="학습에 사용되는 장치")
    parser.add_argument("--model", dest="model_name", default="efficientnet", help="학습에 사용되는 모델")
    parser.add_argument("--image_size", type=int, default=256, help="이미지 resize 크기")
    parser.add_argument("--batch_size", type=int, default=64, help="훈련 배치 사이즈 크기")
    args = parser.parse_args()

    test(args)