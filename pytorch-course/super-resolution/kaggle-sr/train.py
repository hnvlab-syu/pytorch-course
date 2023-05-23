import os
import random
import shutil

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchmetrics import PeakSignalNoiseRatio

from src.dataset import KaggleSRDataset
from src.model import EDSR
from src.utils import split_dataset


def visualize_dataset(
    lr_dir: os.PathLike,
    hr_dir: os.PathLike,
    csv_path: os.PathLike,
    save_dir: os.PathLike,
    n_images: int = 10,
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

    dataset = KaggleSRDataset(
        lr_dir=lr_dir,
        hr_dir=hr_dir,
        csv_path=csv_path,
        transform=transforms.ToTensor()
    )

    indices = random.choices(range(len(dataset)), k=n_images)
    for i in tqdm(indices):
        lr_image, hr_image, image_id = dataset[i]
        _, lr_h, lr_w = lr_image.shape
        _, hr_h, hr_w = hr_image.shape

        background = torch.ones_like(hr_image)
        edge_width = (hr_w - lr_w) // 2
        edge_height = (hr_h - lr_h) // 2
        background[:, edge_height:edge_height+lr_h, edge_width:edge_width+lr_w] = lr_image
        background = (background * 255.0).type(torch.uint8)
        hr_image = (hr_image * 255.0).type(torch.uint8)
        
        _, axs = plt.subplots(ncols=2, squeeze=False)
        for i, img in enumerate([background, hr_image]):
            img = img.detach()
            img = TF.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f'{image_id}.jpg'), dpi=150, bbox_inches='tight', pad_inches=0)
        plt.clf()
        plt.close()


def train_one_epoch(dataloader: DataLoader, device: str, model: nn.Module, loss_fn: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    size = len(dataloader.dataset)
    model.train()
    for batch, (lr_images, hr_images, _) in enumerate(dataloader):
        lr_images = lr_images.to(device)
        hr_images = hr_images.to(device)

        preds = model(lr_images)
        loss = loss_fn(preds, hr_images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 2 == 0:
            loss = loss.item()
            current = batch * len(lr_images)
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
        for lr_images, hr_images, _ in dataloader:
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)

            preds = model(lr_images)

            test_loss += loss_fn(preds, hr_images).item()
            metric.update(preds, hr_images)
    test_loss /= num_batches
    psnr = metric.compute()
    print(f'Test Error: \n PSNR: {psnr:>0.1f}, Avg loss: {test_loss:>8f} \n')

    metric.reset()
    print()


def train() -> None:
    """학습/추론 파이토치 파이프라인입니다.

    :param batch_size: 학습 및 추론 데이터셋의 배치 크기
    :type batch_size: int
    :param epochs: 전체 학습 데이터셋을 훈련하는 횟수
    :type epochs: int
    """
    lr_dir = 
    hr_dir = 
    train_csv_path = 
    test_csv_path = 

    batch_size = 32
    epochs = 5
    lr = 1e-3

    split_dataset(hr_dir)

    visualize_dataset(lr_dir, hr_dir, train_csv_path, save_dir='examples/kaggle-sr/train')
    visualize_dataset(lr_dir, hr_dir, test_csv_path, save_dir='examples/kaggle-sr/test')

    training_data = KaggleSRDataset(
        lr_dir=lr_dir,
        hr_dir=hr_dir,
        csv_path=train_csv_path,
        transform=transforms.ToTensor()
    )

    test_data = KaggleSRDataset(
        lr_dir=lr_dir,
        hr_dir=hr_dir,
        csv_path=test_csv_path,
        transform=transforms.ToTensor()
    )

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=16)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=8)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = EDSR().to(device)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    metric = PeakSignalNoiseRatio().to(device)

    for t in range(epochs):
        print(f'Epoch {t+1}\n-------------------------------')
        train_one_epoch(train_dataloader, device, model, loss_fn, optimizer)
        val_one_epoch(test_dataloader, device, model, loss_fn, metric)
    print('Done!')

    torch.save(model.state_dict(), 'kaggle-sr-edsr.pth')
    print('Saved PyTorch Model State to kaggle-sr-edsr.pth')
