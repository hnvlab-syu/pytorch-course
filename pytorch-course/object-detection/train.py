import argparse
import os
import random
import shutil

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe

import torch
from torch import optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from src.dataset import WheatDataset, collate_fn
from src.model import DetectionModel
from src.utils import split_dataset, MeanAveragePrecision


parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu", type=str, help="device to use for training")
parser.add_argument("--batch_size", default=16, type=int, help="number of samples in a batch")
parser.add_argument("--epochs", default=10, type=int, help="number of epochs to train the model")
parser.add_argument("--lr", default=1e-3, type=float, help="learning rate for the optimizer")
args = parser.parse_args()


def visualize_dataset(image_dir: os.PathLike, csv_path: os.PathLike, save_dir: os.PathLike, n_images: int = 10) -> None:
    """데이터셋 샘플 bbox 그려서 시각화
    
    :param save_dir: bbox 그린 그림 저장할 폴더 경로
    :type save_dir: os.PathLike
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)

    dataset = WheatDataset(
        image_dir=image_dir,
        csv_path=csv_path,
        transform=transforms.ToTensor()
    )

    indexes = random.choices(range(len(dataset)), k=n_images)
    for i in indexes:
        image, target, image_id = dataset[i]
        image = image.numpy().transpose(1, 2, 0)

        plt.imshow(image)
        ax = plt.gca()

        for x1, y1, x2, y2 in target['boxes']:
            w = x2 - x1
            h = y2 - y1

            category_id = 'wheat'

            rect = patches.Rectangle(
                (x1, y1),
                w, h,
                linewidth=1,
                edgecolor='r',
                facecolor='none'
            )
            ax.add_patch(rect)

            text = ax.text(
                x1, y1,
                category_id,
                color='red',
                ha='left',
                va='top',
                fontsize=8
            )
            text.set_path_effects([pe.withStroke(linewidth=3, foreground='black')])

        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f'{image_id}.png'))
        plt.close()


def train_one_epoch(
    dataloader: DataLoader,
    device: str,
    model: torchvision.models.detection,
    optimizer: torch.optim.Optimizer
    ) -> None:
    """Wheat 데이터셋으로 뉴럴 네트워크를 훈련합니다.
    
    :param dataloader: 파이토치 데이터로더
    :type dataloader: DataLoader
    :param device: 훈련에 사용되는 장치
    :type device: str
    :param model: 훈련에 사용되는 모델
    :type model: nn.Module
    :param optimizer: 훈련에 사용되는 옵티마이저
    :type optimizer: torch.optim.Optimizer
    """
    size = len(dataloader.dataset)
    model.train()
    for batch, (images, targets, _) in enumerate(dataloader):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        model.train()
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            current = batch * len(images)
            message = 'total loss: {:.4f}, cls loss: {:>4f}, box loss: {:>4f}, obj loss: {:>4f}, rpn loss: {:>4f}  [{:>5d}/{:>5d}]'
            message = message.format(
                loss.item(),
                loss_dict['loss_classifier'].item(),
                loss_dict['loss_box_reg'].item(),
                loss_dict['loss_objectness'].item(),
                loss_dict['loss_rpn_box_reg'].item(),
                current,
                size
            )
            print(message)


def val_one_epoch(
    dataloader: DataLoader,
    device: str,
    model: torchvision.models.detection,
    metric) -> None:
    """CIFAR-10 데이터셋으로 뉴럴 네트워크의 성능을 테스트합니다.

    :param dataloader: 파이토치 데이터로더
    :type dataloader: DataLoader
    :param device: 테스트에 사용되는 장치
    :type device: _device
    :param model: 테스트에 사용되는 모델
    :type model: nn.Module
    :param loss_fn: 테스트에 사용되는 오차 함수
    :type loss_fn: nn.Module
    """
    num_batches = len(dataloader)
    test_loss = 0
    test_cls_loss = 0
    test_box_loss = 0
    test_obj_loss = 0
    test_rpn_loss = 0
    with torch.no_grad():
        for images, targets, image_ids in dataloader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            model.train()
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            test_loss += loss
            test_cls_loss += loss_dict['loss_classifier']
            test_box_loss += loss_dict['loss_box_reg']
            test_obj_loss += loss_dict['loss_objectness']
            test_rpn_loss += loss_dict['loss_rpn_box_reg']

            model.eval()
            preds = model(images)

            metric.update(preds, image_ids)
    test_loss /= num_batches
    test_cls_loss /= num_batches
    test_box_loss /= num_batches
    test_obj_loss /= num_batches
    test_rpn_loss /= num_batches
    print(f'Test Error: \n Avg loss: {test_loss:>8f} \n Class loss: {test_cls_loss:>8f} \n Box loss: {test_box_loss:>8f} \n Obj loss: {test_obj_loss:>8f} \n RPN loss: {test_rpn_loss:>8f} \n')
    metric.compute()

    metric.reset()
    print()


def train(
    model: torchvision.models.detection,
    device: str = 'cpu', 
    batch_size: int = 16,
    epochs: int = 10,
    lr: float = 1e-3
    ) -> None:
    """학습/추론 파이토치 파이프라인입니다.

    :param batch_size: 학습 및 추론 데이터셋의 배치 크기
    :type batch_size: int
    :param epochs: 전체 학습 데이터셋을 훈련하는 횟수
    :type epochs: int
    """
    csv_path = 'data/global-wheat-detection/train.csv'
    train_image_dir = 'data/global-wheat-detection/train'
    train_csv_path = 'data/global-wheat-detection/train_answer.csv'
    test_csv_path = 'data/global-wheat-detection/test_answer.csv'

    num_classes = 1
    split_dataset(csv_path)
    visualize_dataset(train_image_dir, train_csv_path, save_dir='examples/global-wheat-detection/train')
    visualize_dataset(train_image_dir, test_csv_path, save_dir='examples/global-wheat-detection/test')

    training_data = WheatDataset(
        image_dir=train_image_dir,
        csv_path=train_csv_path,
        transform=transforms.ToTensor()
    )
    test_data = WheatDataset(
        image_dir=train_image_dir,
        csv_path=test_csv_path,
        transform=transforms.ToTensor(),
    )

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=int(batch_size/2), num_workers=0, collate_fn=collate_fn)
    
    model = DetectionModel(num_classes=num_classes).make_model().to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.005)
    metric = MeanAveragePrecision(csv_path=test_csv_path)

    for t in range(epochs):
        print(f'Epoch {t+1}\n-------------------------------')
        train_one_epoch(train_dataloader, device, model, optimizer)
        val_one_epoch(test_dataloader, device, model, metric)
    print('Done!')

    torch.save(model.state_dict(), 'wheat-faster-rcnn.pth')
    print('Saved PyTorch Model State to wheat-faster-rcnn.pth')



if __name__ == '__main__':
    train(model=DetectionModel, device=args.device, batch_size=args.batch_size, \
          epochs=args.epochs, lr=args.lr)