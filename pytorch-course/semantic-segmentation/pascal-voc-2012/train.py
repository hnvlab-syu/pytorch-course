import os
import random
import shutil
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import draw_segmentation_masks
from torchvision.models.segmentation import deeplabv3_resnet50
from torchmetrics.classification import MulticlassJaccardIndex

from src.dataset import PascalVOC2012Dataset
from src.utils import split_dataset, get_transform, get_mask_transform


def visualize_dataset(
    image_dir: os.PathLike,
    label_dir: os.PathLike,
    csv_path: os.PathLike,
    size: Sequence[int],
    save_dir: os.PathLike,
    n_images: int = 10,
    alpha: float = 0.5
) -> None:
    """데이터셋 샘플 bbox 그려서 시각화
    
    :param save_dir: bbox 그린 그림 저장할 폴더 경로
    :type save_dir: os.PathLike
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)

    dataset = PascalVOC2012Dataset(
        image_dir=image_dir,
        label_dir=label_dir,
        csv_path=csv_path,
        transform=get_transform(size=size),
        mask_transform=get_mask_transform(size=size)
    )

    indices = random.choices(range(len(dataset)), k=n_images)
    for i in tqdm(indices):
        image, target, meta_data = dataset[i]
        image = (image * 255.0).type(torch.uint8)

        result = draw_segmentation_masks(image, target.type(torch.bool), alpha=alpha)
        plt.imshow(result.permute(1, 2, 0).numpy())

        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f"{meta_data['image_id']}.jpg"), dpi=150, bbox_inches='tight', pad_inches=0)
        plt.clf()


def train_one_epoch(dataloader: DataLoader, device: str, model: nn.Module, loss_fn: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    size = len(dataloader.dataset)
    model.train()
    for batch, (images, targets, _) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device)

        preds = model(images)['out']
        preds = torch.softmax(preds, dim=1)
        loss = loss_fn(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss = loss.item()
            current = batch * len(images)
            print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')


def val_one_epoch(dataloader: DataLoader, device: str, model: nn.Module, loss_fn: nn.Module, metric) -> None:
    """Dirty-MNIST 데이터셋으로 뉴럴 네트워크의 성능을 테스트합니다.

    :param dataloader: 파이토치 데이터로더
    :type dataloader: DataLoader
    :param device: 훈련에 사용되는 장치
    :type device: str
    :param model: 훈련에 사용되는 모델
    :type model: nn.Module
    :param loss_fn: 훈련에 사용되는 오차 함수
    :type loss_fn: nn.Module
    """
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for images, targets, _ in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            preds = model(images)['out']
            preds = torch.softmax(preds, dim=1)

            test_loss += loss_fn(preds, targets).item()
            metric.update(preds, targets.argmax(dim=1))
    test_loss /= num_batches
    miou = metric.compute()
    print(f'Test Error: \n mIoU: {(100*miou):>0.1f}, Avg loss: {test_loss:>8f} \n')
    
    metric.reset()
    print()


def train() -> None:
    """학습/추론 파이토치 파이프라인입니다.

    :param batch_size: 학습 및 추론 데이터셋의 배치 크기
    :type batch_size: int
    :param epochs: 전체 학습 데이터셋을 훈련하는 횟수
    :type epochs: int
    """
    image_dir = 
    label_dir = 
    train_csv_path = 
    test_csv_path = 
    
    size = (500, 500)
    num_classes = 20
    batch_size = 32
    epochs = 5
    lr = 1e-3


    split_dataset(label_dir)

    visualize_dataset(image_dir, label_dir, train_csv_path, size, save_dir='examples/pascal-voc-2012/train', alpha=0.8)
    visualize_dataset(image_dir, label_dir, test_csv_path, size, save_dir='examples/pascal-voc-2012/test', alpha=0.8)

    training_data = PascalVOC2012Dataset(
        image_dir=image_dir,
        label_dir=label_dir,
        csv_path=train_csv_path,
        transform=get_transform(size=size),
        mask_transform=get_mask_transform(size=size)
    )

    test_data = PascalVOC2012Dataset(
        image_dir=image_dir,
        label_dir=label_dir,
        csv_path=test_csv_path,
        transform=get_transform(size=size),
        mask_transform=get_mask_transform(size=size)
    )

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=16)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=8)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = deeplabv3_resnet50(num_classes=num_classes+1).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    metric = MulticlassJaccardIndex(num_classes=num_classes+1, ignore_index=0).to(device)

    for t in range(epochs):
        print(f'Epoch {t+1}\n-------------------------------')
        train_one_epoch(train_dataloader, device, model, loss_fn, optimizer)
        val_one_epoch(test_dataloader, device, model, loss_fn, metric)
    print('Done!')

    torch.save(model.state_dict(), 'pascal-voc-2012-deeplabv3.pth')
    print('Saved PyTorch Model State to pascal-voc-2012-deeplabv3.pth')
