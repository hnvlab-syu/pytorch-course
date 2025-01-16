import os
import glob
import argparse

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from src.model import create_model
from src.dataset import COCODataset
from src.utils import set_seed, get_transform, split_dataset, collate_fn, SEED


def test(segmentation_model, data, batch_size, ckpt, device, num_workers):
    num_classes = 91

    image_list = glob.glob(os.path.join(data, 'val2017/*.jpg'))
    _, _, test_list = split_dataset(image_list)

    test_data = COCODataset(
        os.path.join(data, 'val2017'),
        os.path.join(data, 'instances_val2017.json'),
        image_list=test_list,
        transform=get_transform(subject='train')
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    model = create_model(segmentation_model, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(ckpt))

    test_map = MeanAveragePrecision()

    # test
    model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(test_dataloader):
            inputs = [image.to(device) for image in inputs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(inputs)
            test_map.update(outputs, targets)

    test_mAP = test_map.compute()
    test_map.reset()

    print(test_mAP)


if __name__ == '__main__':
    set_seed(SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--num_workers', type=int, default=0, help='number of worker processes for data loading')
    parser.add_argument('-m', '--model', type=str, default='mask_rcnn')
    parser.add_argument('-d', '--data_path', dest='data', type=str, default='../dataset')
    parser.add_argument('-b', '--batch_size', dest='batch', type=int, default=8)
    parser.add_argument('-c', '--checkpoint_path', dest='ckpt', type=str, default='./checkpoint/')
    parser.add_argument('-dc', '--device', type=str, default='cuda')
    args = parser.parse_args()
    
    test(args.model, args.data, args.batch, args.ckpt, args.device, args.num_workers)