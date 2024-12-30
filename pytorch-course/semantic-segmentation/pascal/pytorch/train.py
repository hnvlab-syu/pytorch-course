import os
import argparse

import wandb
from sklearn.model_selection import train_test_split

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchmetrics

from src.dataset import PascalVOCDataset
from src.model import create_model
from src.utils import set_seed, get_transform, collate_fn, visualize_batch, SEED


def main(model_name, data_dir, batch_size, epochs, save_path, device, num_workers):
    wandb.init(project="pascal-semanticsegmentation")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(os.path.join(data_dir, 'ImageSets', 'Segmentation', 'train.txt'), 'r') as f:
        train_list = f.read().splitlines()
    train_list, valid_list = train_test_split(train_list, train_size=0.8, shuffle=True, random_state=SEED)

    train_data = PascalVOCDataset(
        data_dir=data_dir,
        image_list=train_list,
        image_transform=get_transform(subject='image'),
        mask_transform=get_transform(subject='mask')
    )
    valid_data = PascalVOCDataset(
        data_dir=data_dir,
        image_list=valid_list,
        image_transform=get_transform(subject='image'),
        mask_transform=get_transform(subject='mask')
    )

    train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)

    model = create_model(model=model_name).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    num_classes = 21
    train_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=num_classes).to(device)
    valid_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=num_classes).to(device)

    for t in range(epochs):
        print(f'Epoch {t+1}\n-------------------------------')
        # training
        size = len(train_dataloader.dataset)
        model.train()
        for batch, (inputs, target) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            target = target.to(device)

            output = model(inputs)['out']
            loss = loss_fn(output, target)

            predictions = torch.argmax(output, dim=1)
            train_iou.update(predictions, target)

            # visualize_batch(inputs, target, predictions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 10 == 0:
                loss = loss.item()
                current = batch * len(inputs)
                print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')

        train_miou = train_iou.compute()

        wandb.log({
            "epoch": t + 1,
            "train_loss": loss,
            "train_miou": train_miou
        })

        train_iou.reset()

        # validation
        num_batches = len(valid_dataloader)
        total_valid_loss = 0
        best_valid_miou = 0
        model.eval()
        with torch.no_grad():
            for inputs, target in valid_dataloader:
                inputs = inputs.to(device)
                target = target.to(device)

                output = model(inputs)['out']
                valid_loss = loss_fn(output, target)

                predictions = torch.argmax(output, dim=1)
                valid_iou.update(predictions, target)
                
                # visualize_batch(inputs, target, predictions)

                total_valid_loss += valid_loss.item()

        valid_miou = valid_iou.compute()
        total_valid_loss /= num_batches
        print(f'Valid Error: \n mIoU: {(valid_miou):>0.3f}, Avg loss: {total_valid_loss:>8f} \n')
    
        wandb.log({
            "epoch": t + 1,
            "val_loss": total_valid_loss,
            "valid_miou": valid_miou
        })

        if best_valid_miou < valid_miou:
            best_valid_miou = valid_miou
            torch.save(model.state_dict(), f'{save_path}/best_epoch-pascalvoc-{model_name}.pth')
            print(f'{t+1} epoch: Saved Model State to best_epoch-pascalvoc-{model_name}.pth\n')
        valid_iou.reset()

    torch.save(model.state_dict(), f'{save_path}/last_epoch-pascalvoc-{model_name}.pth')
    print(f'Saved Model State to last_epoch-pascalvoc-{model_name}.pth\n')

    print('Done!')
    wandb.finish()


if __name__ == '__main__':
    set_seed(SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--num_workers', type=int, default=16, help='number of worker processes for data loading')
    parser.add_argument('-m', '--model', type=str, default='deeplabv3')
    parser.add_argument('-b', '--batch_size', dest='batch', type=int, default=32)
    parser.add_argument('-e', '--epoch', type=int, default=10)
    parser.add_argument('-d', '--data_path', dest='data', type=str, default='../dataset/VOC2012')
    parser.add_argument('-s', '--save_path', dest='save', type=str, default='./checkpoint/')
    parser.add_argument('-dc', '--device', type=str, default='cuda')
    args = parser.parse_args()
    
    main(args.model, args.data, args.batch, args.epoch, args.save, args.device, args.num_workers)