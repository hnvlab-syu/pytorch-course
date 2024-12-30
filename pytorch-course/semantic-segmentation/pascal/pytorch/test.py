import os
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchmetrics

from src.dataset import PascalVOCDataset
from src.model import create_model
from src.utils import set_seed, get_transform, collate_fn, visualize_batch, SEED


def main(model_name, data_dir, batch_size, device, ckpt, num_workers):
    with open(os.path.join(data_dir, 'ImageSets', 'Segmentation', 'val.txt'), 'r') as f:
        test_list = f.read().splitlines()
    
    test_data = PascalVOCDataset(
        data_dir=data_dir,
        image_list=test_list,
        image_transform=get_transform(subject='image'),
        mask_transform=get_transform(subject='mask')
    )
    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)

    model = create_model(model=model_name).to(device)
    model.load_state_dict(torch.load(ckpt))
    loss_fn = nn.CrossEntropyLoss()

    num_classes = 21
    test_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=num_classes).to(device)

    # test
    num_batches = len(test_dataloader)
    total_test_loss = 0
    model.eval()
    with torch.no_grad():
        for inputs, target in test_dataloader:
            inputs = inputs.to(device)
            target = target.to(device)

            output = model(inputs)['out']
            test_loss = loss_fn(output, target)

            predictions = torch.argmax(output, dim=1)
            test_iou.update(predictions, target)
            
            # visualize_batch(inputs, target, predictions)

            total_test_loss += test_loss.item()

    test_miou = test_iou.compute()
    total_test_loss /= num_batches
    print(f'Valid Error: \n mIoU: {(test_miou):>0.3f}, Avg loss: {total_test_loss:>8f} \n')
    test_iou.reset()

    print('Done!')


if __name__ == '__main__':
    set_seed(SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--num_workers', type=int, default=16, help='number of worker processes for data loading')
    parser.add_argument('-m', '--model', type=str, default='deeplabv3')
    parser.add_argument('-b', '--batch_size', dest='batch', type=int, default=32)
    parser.add_argument('-d', '--data_path', dest='data', type=str, default='../dataset/VOC2012')
    parser.add_argument('-dc', '--device', type=str, default='cuda')
    parser.add_argument('-c', '--ckpt_path', dest='ckpt', type=str, default='./checkpoint/')
    args = parser.parse_args()
    
    main(args.model, args.data, args.batch, args.device, args.ckpt, args.num_workers)