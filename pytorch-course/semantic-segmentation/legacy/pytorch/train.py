import os
import random
import shutil
from typing import Sequence
import warnings

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

# 경고 메세지 무시
warnings.filterwarnings("ignore")

# 데이터셋 시각화
def visualize_dataset(
        image_dir: os.PathLike,
        label_dir: os.PathLike,
        csv_path: os.PathLike,
        num_classes: int,
        size: Sequence[tuple],
        save_dir: os.PathLike,
        n_images: int = 10,
        alpha: float = 0.7
) -> None:
    """데이터셋 segmentation 그려서 시각화
    
    :param image_dir: 원본 이미지 디렉토리
    :type image_dir: os.PathLike
    :param label_dir: 정답 레이블 이미지 폴더 경로
    :type label_dir: os.PathLike
    :param csv_path: csv 폴더 경로
    :type csv_path: os.PathLike
    :param num_classes: 클래스 개수
    :type num_classes: int
    :param size: resize할 이미지 크기
    :type size: Sequence[tuple]
    :param save_dir: segmentation 그린 그림 저장할 폴더 경로
    :type save_dir: os.PathLike
    :param n_images: segmentation 그릴 그림 개수
    :type n_images: int
    :param alpha: segmentation 투명도
    :type alpha: float
    """
    # 디렉토리 재생성
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)

    # 데이터셋 초기화
    dataset = PascalVOC2012Dataset(
        image_dir=image_dir,
        label_dir=label_dir,
        csv_path=csv_path,
        num_classes=num_classes,
        transform=get_transform(size=size),
        mask_transform=get_mask_transform(size=size)
    )

    # 랜덤 이미지 시각화
    indices = random.choices(range(len(dataset)), k=n_images)
    for i in tqdm(indices):

        # 데이터 가져오기
        image, target, meta_data = dataset[i]
        image = (image * 255.0).type(torch.uint8)

        # 세그멘테이션 마스크 시각화
        result = draw_segmentation_masks(image, target.type(torch.bool), alpha=alpha)
        plt.imshow(result.permute(1, 2, 0).numpy())

        # 저장 및 초기화
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f"{meta_data['image_id']}.jpg"), dpi=150, bbox_inches='tight', pad_inches=0)
        plt.clf()


# 훈련 epoch
def train_one_epoch(dataloader: DataLoader, device: str, model: nn.Module, loss_fn: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    size = len(dataloader.dataset)
    model.train()
    for batch, (images, targets, _) in enumerate(dataloader):

        # images, targets 이동
        images = images.to(device)
        targets = targets.to(device)

        # 모델 예측
        preds = model(images)['out']

        # loss 계산
        preds = torch.softmax(preds, dim=1)
        loss = loss_fn(preds, targets)
        
        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 진행도 출력
        if batch % 10 == 0:
            loss = loss.item()
            current = batch * len(images)
            print(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')


# 검증 epoch
def val_one_epoch(dataloader: DataLoader, device: str, model: nn.Module, loss_fn: nn.Module, metric) -> None:
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for images, targets, _ in dataloader:
            # images, targets 이동
            images = images.to(device)
            targets = targets.to(device)

            # 모델 예측
            preds = model(images)['out']

            # loss 계산
            preds = torch.softmax(preds, dim=1)
            test_loss += loss_fn(preds, targets).item()
            metric.update(preds, targets.argmax(dim=1))

        # metric 계산
        test_loss /= num_batches
        miou = metric.compute()
        print(f'Test Error: \n mIoU: {(100*miou):>0.3f}, Avg loss: {test_loss:>8f}\n')

        # metric 초기화
        metric.reset()
        print()


def train() -> None:
    # 디렉토리 설정
    image_dir = 'data/VOC2012/JPEGImages'
    label_dir = 'data/VOC2012/SegmentationClass'
    train_csv_path = 'data/VOC2012/train_answer.csv'
    test_csv_path = 'data/VOC2012/test_answer.csv'
    
    # 하이퍼파라미터 설정
    size = (500, 500)
    num_classes = 20
    batch_size = 8
    epochs = 10
    lr = 1e-3

    # 데이터셋 분리
    split_dataset(label_dir)

    # 데이터셋 시각화
    print("Dataset Visualize")
    visualize_dataset(image_dir, label_dir, train_csv_path, num_classes, size, save_dir='examples/train', alpha=0.7)
    visualize_dataset(image_dir, label_dir, test_csv_path, num_classes, size, save_dir='examples/test', alpha=0.7)
    print("Done\n")

    # train/test 데이터셋 초기화
    training_data = PascalVOC2012Dataset(
        image_dir=image_dir,
        label_dir=label_dir,
        csv_path=train_csv_path,
        num_classes=num_classes,
        transform=get_transform(size=size),
        mask_transform=get_mask_transform(size=size)
    )    
    test_data = PascalVOC2012Dataset(
        image_dir=image_dir,
        label_dir=label_dir,
        csv_path=test_csv_path,
        num_classes=num_classes,
        transform=get_transform(size=size),
        mask_transform=get_mask_transform(size=size)
    )

    # 데이터로더 초기화
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=32)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=8)

    # device 초기화
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # model 초기화
    model = deeplabv3_resnet50(pretrained=True, num_classes=num_classes+1)
    model.to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    metric = MulticlassJaccardIndex(num_classes=num_classes+1, ignore_index=0).to(device)

    # epoch 진행
    print(f"<<<   Training by {device}   >>>\n")
    for t in range(1, epochs+1):
        print(f"Epoch {t}\n-------------------------------")
        train_one_epoch(train_dataloader, device, model, loss_fn, optimizer)
        val_one_epoch(test_dataloader, device, model, loss_fn, metric)
        if t % 10 == 0 or t == 1:
            torch.save(model.state_dict(), 'pascal-voc-2012-pretrained-deeplabv3.pth')
            print('Saved PyTorch Model State to pascal-voc-2012-pretrained-deeplabv3.pth\n')
    print("Done")

if __name__ == "__main__":
    train()