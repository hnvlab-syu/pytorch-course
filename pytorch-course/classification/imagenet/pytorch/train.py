import argparse

import wandb
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.dataset import ImagenetDataset, get_transform
from src.model import create_model
from src.utils import rename_dir, split_dataset, set_seed


def main(args):
    wandb.init(project="imagenet-classification")

    batch_size = args.batch_size
    epochs = args.epochs
    lr = 1e-3
    device = args.device
    image_size = args.image_size
    model_name = args.model_name

    data_dir = '../dataset'
    class_txt = '../dataset/folder_num_class_map.txt'
    _ = rename_dir(txt_path=class_txt, data_dir=data_dir)
    train_x, train_y, val_x, val_y, _, _ = split_dataset(data_dir=data_dir)

    train_data = ImagenetDataset(
        image_dir=train_x,
        class_name=train_y,
        transform=get_transform(image_size=image_size)
    )
    valid_data = ImagenetDataset(
        image_dir=val_x,
        class_name=val_y,
        transform=get_transform(image_size=image_size)
    )
    print("Initialize Dataset\n")

    train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=0)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, num_workers=0)
    print("Initialize DataLoader\n")

    model = create_model(model=model_name).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    print("Initialize Pytorch Model\n")

    best_accuracy = 0

    for t in range(epochs):
        print(f'Epoch {t+1}\n-------------------------------')
        # training
        size = len(train_dataloader.dataset)
        model.train()
        for batch, (images, targets) in enumerate(train_dataloader):
            images = images.to(device)
            targets = targets.to(device)
            targets = torch.flatten(targets)

            preds = model(images)
            loss = loss_fn(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss = loss.item()
                current = batch * len(images)
                print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')

            wandb.log({
                "epoch": t + 1,
                "train_loss": loss,
            })

        # validation
        size = len(valid_dataloader.dataset)
        num_batches = len(valid_dataloader)
        model.eval()
        total_valid_loss = 0
        correct = 0
        with torch.no_grad():
            for images, targets in valid_dataloader:
                images = images.to(device)
                targets = targets.to(device)
                targets = torch.flatten(targets)

                preds = model(images)
                valid_loss = loss_fn(preds, targets)

                total_valid_loss += valid_loss.item()
                correct += (preds.argmax(1) == targets).float().sum().item()

                wandb.log({
                    "epoch": t + 1,
                    "val_loss": valid_loss.item(),
                })

        total_valid_loss /= num_batches
        correct /= size
        print(f'Valid Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {total_valid_loss:>8f} \n')
    
        wandb.log({
            "epoch": t + 1,
            "valid_accuracy": correct * 100
        })

        if best_accuracy < correct:
            best_accuracy = correct
            torch.save(model.state_dict(), f'best_epoch-imagenet-{model_name}.pth')
            print(f'{t+1} epoch: Saved Model State to best_epoch-imagenet-{model_name}.pth\n')

    torch.save(model.state_dict(), f'last_epoch-imagenet-{model_name}.pth')
    print(f'Saved Model State to last_epoch-imagenet-{model_name}.pth\n')

    print('Done!')
    wandb.finish()


if __name__ == '__main__':
    set_seed(36)

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", help="학습에 사용되는 장치")
    parser.add_argument("--image_size", type=int, default=256, help="이미지 resize 크기")
    parser.add_argument("--batch_size", type=int, default=64, help="훈련 배치 사이즈 크기")
    parser.add_argument("--epochs", type=int, default=30, help="훈련 에폭 크기")
    parser.add_argument("--model", dest='model_name', type=str, default='efficientnet', help="사용할 모델 선택")
    args = parser.parse_args()

    main(args)