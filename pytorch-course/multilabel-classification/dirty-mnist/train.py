import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics.classification import MultilabelF1Score

from src.dataset import DirtyMnistDataset
from src.model import MultiLabelResNet
from src.utils import split_dataset, get_train_transform, get_val_transform


def train_one_epoch(dataloader: DataLoader, device: str, model: nn.Module, loss_fn: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    """Dirty-MNIST 데이터셋으로 뉴럴 네트워크를 훈련합니다.
    
    :param dataloader: 파이토치 데이터로더
    :type dataloader: DataLoader
    :param device: 훈련에 사용되는 장치
    :type device: str
    :param model: 훈련에 사용되는 모델
    :type model: nn.Module
    :param loss_fn: 훈련에 사용되는 오차 함수
    :type loss_fn: nn.Module
    :param optimizer: 훈련에 사용되는 옵티마이저
    :type optimizer: torch.optim.Optimizer
    """
    size = len(dataloader.dataset)
    model.train()
    for batch, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device)

        preds = model(images)
        # print(targets, preds)
        loss = loss_fn(preds, targets.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
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
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            preds = model(images)
            # print(targets, preds)

            test_loss += loss_fn(preds, targets.float()).item()
            # print(torch.sigmoid(preds) >= 0.5)
            # print(targets)
            metric.update(torch.sigmoid(preds), targets)
    test_loss /= num_batches
    f1_score = metric.compute()
    print(f'Test Error: \n F1 Score: {(100*f1_score):>0.1f}%, Avg loss: {test_loss:>8f} \n')


def train() -> None:
    """학습/추론 파이토치 파이프라인입니다.

    :param batch_size: 학습 및 추론 데이터셋의 배치 크기
    :type batch_size: int
    :param epochs: 전체 학습 데이터셋을 훈련하는 횟수
    :type epochs: int
    """
    csv_path = 
    image_dir = 
    train_csv_path = 
    test_csv_path = 

    num_classes = 26
    batch_size = 32
    epochs = 5
    lr = 1e-3

    split_dataset(csv_path)

    training_data = DirtyMnistDataset(
        image_dir=image_dir,
        csv_path=train_csv_path,
        transform=get_train_transform()
    )
    test_data = DirtyMnistDataset(
        image_dir=image_dir,
        csv_path=test_csv_path,
        transform=get_val_transform()
    )

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=16)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=8)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MultiLabelResNet(num_classes=num_classes).to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    metric = MultilabelF1Score(num_labels=num_classes).to(device)

    for t in range(epochs):
        print(f'Epoch {t+1}\n-------------------------------')
        train_one_epoch(train_dataloader, device, model, loss_fn, optimizer)
        val_one_epoch(test_dataloader, device, model, loss_fn, metric)
    print('Done!')

    torch.save(model.state_dict(), 'dirty-mnist-resnet.pth')
    print('Saved PyTorch Model State to dirty-mnist-resnet.pth')
