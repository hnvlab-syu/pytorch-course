import os
import glob
import argparse

import wandb

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from src.model import create_model
from src.dataset import COCODataset
from src.utils import set_seed, get_transform, split_dataset, collate_fn, SEED


def train(segmentation_model, data, batch_size, epochs, save, device, num_workers):
    wandb.init(project='instance-semanticsegmentation')

    if not os.path.exists(save):
        os.makedirs(save)

    image_list = glob.glob(os.path.join(data, 'val2017/*.jpg'))
    train_list, val_list, _ = split_dataset(image_list)

    train_data = COCODataset(
        os.path.join(data, 'val2017'),
        os.path.join(data, 'instances_val2017.json'),
        image_list=train_list,
        transform=get_transform(subject='train')
    )
    val_data = COCODataset(
        os.path.join(data, 'val2017'),
        os.path.join(data, 'instances_val2017.json'),
        image_list=val_list,
        transform=get_transform(subject='train')
    )

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    num_classes = 91
    model = create_model(segmentation_model, num_classes=num_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    val_map = MeanAveragePrecision()
    best_val_map = 0

    for t in range(epochs):
        print(f'Epoch {t+1}\n-------------------------------')
        
        #training
        size = len(train_dataloader.dataset)
        model.train()
        for batch, (inputs, targets) in enumerate(train_dataloader):
            inputs = [image.to(device) for image in inputs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(inputs, targets)

            cls_loss = outputs.get('loss_classifier', 0)
            box_loss = outputs.get('loss_box_reg', 0)
            mask_loss = outputs.get('loss_mask', 0)

            loss = cls_loss + box_loss + mask_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 20 == 0:
                current = batch * len(inputs)
                message = 'total loss: {:.4f}, cls loss: {:.4f}, box loss: {:.4f}, mask loss: {:.4f}  [{:>5d}/{:>5d}]'
                message = message.format(
                    loss,
                    cls_loss,
                    box_loss,
                    mask_loss,
                    current,
                    size
                )
                print(message)

        wandb.log({
            "epoch": t + 1,
            "train-cls_loss": cls_loss,
            "train-box_loss": box_loss,
            "train-mask_loss": mask_loss,
            "train-total_loss": loss
        })
        
        # validation
        model.eval()
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                inputs = [image.to(device) for image in inputs]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                outputs = model(inputs)
                val_map.update(outputs, targets)

        val_mAP = val_map.compute()
        val_map.reset()

        if best_val_map < val_mAP['map'].item():
            best_val_map = val_mAP['map'].item()
            torch.save(model.state_dict(), f'{save}/best-instance-segmentation-{segmentation_model}.pth')
            print(f'Saved PyTorch Model State to best-instance-segmentation-{segmentation_model}.pth')

        wandb.log({"epoch": t + 1, "val_mAP": val_mAP['map'].item()})

    torch.save(model.state_dict(), f'{save}/last-instance-segmentation-{segmentation_model}.pth')
    print(f'Saved PyTorch Model State to last-instance-segmentation-{segmentation_model}.pth')


if __name__ == '__main__':
    set_seed(SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--num_workers', type=int, default=0, help='number of worker processes for data loading')
    parser.add_argument('-m', '--model', type=str, default='mask_rcnn')
    parser.add_argument('-d', '--data_path', dest='data', type=str, default='../dataset')
    parser.add_argument('-b', '--batch_size', dest='batch', type=int, default=8)
    parser.add_argument('-e', '--epoch', type=int, default=10)
    parser.add_argument('-s', '--save_path', dest='save', type=str, default='./checkpoint/')
    parser.add_argument('-dc', '--device', type=str, default='cuda')
    args = parser.parse_args()
    
    train(args.model, args.data, args.batch, args.epoch, args.save, args.device, args.num_workers)