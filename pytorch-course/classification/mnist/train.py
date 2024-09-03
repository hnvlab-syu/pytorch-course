import argparse

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.dataset import get_mnist
from src.model import NeuralNetwork


parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu", help="학습에 사용되는 장치")
args = parser.parse_args()


# 학습 에포크
def train_one_epoch(dataloader: DataLoader, device: str, model: nn.Module, loss_fn: nn.Module, optimizer) -> None:
    """MNIST 데이터셋으로 뉴럴 네트워크 훈련
    
    param dataloader: 파이토치 데이터로더
    param dataloader: DataLoader
    param device: 훈련에 사용되는 장치
    param device: str
    param model: 훈련에 사용되는 모델
    param model: nn.Module
    param loss_fn: 훈련에 사용되는 오차함수
    param loss_fn: nn.Module
    param optimizer: 훈련에 사용되는 옵티마이저
    param optimizer: torch.optim.Optimizer
    """

    size = len(dataloader.dataset)
    model.train()
    # 훈련
    for batch, (images, targets) in enumerate(dataloader):
        # 데이터 gpu로 이동
        images = images.to(device)
        targets = targets.to(device)
        targets = torch.flatten(targets)

        # 예측 및 오차 계산
        preds = model(images)
        loss = loss_fn(preds, targets)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 학습 과정 출력
        if batch % 100 == 0:
            loss = loss.item()
            current = batch * len(images)
            print(f'loss: {loss:>7f} [{current:5d}/{size:>5d}]')


def valid_one_epoch(dataloader: DataLoader, device: str, model: nn.Module, loss_fn: nn.Module) -> None:
    """MNIST 데이터셋으로 뉴럴 네트워크 성능 테스트
    
    param dataloader: 파이토치 데이터로더
    param dataloader: DataLoader
    param device: 훈련에 사용되는 장치
    param device: str
    param model: 훈련에 사용되는 모델
    param model: nn.Module
    param loss_fn: 훈련에 사용되는 오차함수
    param loss_fn: nn.Module
    """

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    correct = 0
    # 검증
    with torch.no_grad():
        for images, targets in dataloader:
            # 데이터 gpu로 이동
            images = images.to(device)
            targets = targets.to(device)
            targets = torch.flatten(targets)

            # 예측
            preds = model(images)
            
            # 결과값 합산
            test_loss += loss_fn(preds, targets).item()
            correct += (preds.argmax(1) == targets).float().sum().item()

    # 평균 계산 및 출력
    test_loss /= num_batches
    correct /= size
    print(f'TestError: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n')


def train(device: str):
    # 하이퍼파라미터 값 설정
    num_classes = 10
    batch_size = 32
    epochs = 5
    lr = 1e-3

    """학습/추론 파이토치 파이프라인
    
    param batch_size: 학습 및 추론 데이터셋의 배치 크기
    type batch_size: int
    param epochs: 전체 학습 데이터셋 훈련 횟수
    type epochs: int
    """

    data_dir = 'data'
    train_data, test_data = get_mnist(data_dir)
    # print("train_data len:", len(train_data))
    # print("test_data len:", len(test_data))
    # print("data[0]:", len(train_data[0]), "(img, label)")
    # print("img:", len(train_data[0][0]), "(img channel)")
    # print("img shape:", train_data[0][0][0].shape, "(H, W)")
    # print("label:", train_data[0][1], "(label)")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    model = NeuralNetwork(num_classes=num_classes).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # 학습
    for t in range(epochs):
        print(f'Epoch {t+1}\n-------------------------')
        train_one_epoch(train_loader, device, model, loss_fn, optimizer)
        valid_one_epoch(test_loader, device, model, loss_fn)
    print("Done!")

    # 결과 저장
    torch.save(model.state_dict(), 'mnist-net.pth')
    print('Saved Pytorch Model State to mnist-net.pth')

if __name__ == '__main__':
    train(device=args.device)