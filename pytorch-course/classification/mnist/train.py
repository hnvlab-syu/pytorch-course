import argparse
import wandb

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn import functional as F

from src.dataset import get_mnist
from src.model import NeuralNetwork


parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu", help="학습에 사용되는 장치")
args = parser.parse_args()


# 테스트 이미지 배치에 대한 예측을 로그하기 위한 편리한 함수
def log_test_predictions(images, labels, outputs, predicted, test_table, log_counter):
    # 모든 클래스에 대한 신뢰도 점수 얻기
    scores = F.softmax(outputs.data, dim=1)
    log_scores = scores.cpu().numpy()
    log_images = images.cpu().numpy()
    log_labels = labels.cpu().numpy()
    log_preds = predicted.cpu().numpy()
    # 이미지 순서에 따라 id 추가하기
    _id = 0
    for i, l, p, s in zip(log_images, log_labels, log_preds, log_scores):
        # 데이터 테이블에 필요한 정보 추가하기:
        # id, 이미지 픽셀, 모델의 추측, 진짜 라벨, 모든 클래스에 대한 점수
        img_id = str(_id) + "_" + str(log_counter)
        test_table.add_data(img_id, wandb.Image(i), p, l, *s)
        _id += 1


def train_one_epoch(dataloader: DataLoader, device: str, model: nn.Module, loss_fn: nn.Module, optimizer, epoch: int) -> None:
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
    
    for batch, (images, targets) in enumerate(dataloader):
        
        images = images.to(device)
        targets = targets.to(device)
        targets = torch.flatten(targets)

        preds = model(images)
        loss = loss_fn(preds, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            wandb.log({"train_loss": loss, "epoch": epoch})
            loss = loss.item()
            current = batch * len(images)
            print(f'loss: {loss:>7f} [{current:5d}/{size:>5d}]')


def valid_one_epoch(dataloader: DataLoader, device: str, model: nn.Module, loss_fn: nn.Module, epoch: int, test_table: wandb.Table) -> None:
    """MNIST 데이터셋으로 뉴럴 네트워크의 성능을 테스트합니다.

    :param dataloader: 파이토치 데이터로더
    :type dataloader: DataLoader
    :param device: 훈련에 사용되는 장치
    :type device: str
    :param model: 훈련에 사용되는 모델
    :type model: nn.Module
    :param loss_fn: 훈련에 사용되는 오차 함수
    :type loss_fn: nn.Module
    """

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for batch, (images, targets) in enumerate(dataloader):
            
            images = images.to(device)
            targets = targets.to(device)
            targets = torch.flatten(targets)

            preds = model(images)
            
            test_loss += loss_fn(preds, targets).item()
            correct += (preds.argmax(1) == targets).float().sum().item()

            if batch == 0:
                log_test_predictions(images, targets, preds, preds.argmax(1), test_table, epoch)

    test_loss /= num_batches
    correct /= size
    print(f'Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n')
    wandb.log({"test_loss": test_loss, "test_accuracy": correct, "epoch": epoch})


def train(device: str):
    # 하이퍼파라미터 값 설정
    num_classes = 10
    batch_size = 32
    epochs = 10
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

    test_table = wandb.Table(columns=["id", "image", "predicted", "true", *[f"class_{i}_score" for i in range(10)]])

    for t in range(epochs):
        print(f'Epoch {t+1}\n-------------------------')
        train_one_epoch(train_loader, device, model, loss_fn, optimizer, t+1)
        valid_one_epoch(test_loader, device, model, loss_fn, t+1, test_table)
    print('Done!')

    wandb.log({"predictions": test_table})
    torch.save(model.state_dict(), 'mnist-net.pth')
    print('Saved Pytorch Model State to mnist-net.pth')


if __name__ == "__main__":
    wandb.init(
        # set the wandb project where this run will be logged
        project="mnist_with_wandb",

        # track hyperparameters and run metadata
        config={
            "learning_rate": 1e-3,
            "architecture": "NeuralNetwork",
            "dataset": "MNIST",
            "epochs": 10,
        }
    )
    train(device=args.device)
    wandb.finish()
