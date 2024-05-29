import argparse
import os
import random
import shutil

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms

from src.dataset import WheatDataset


parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu", help="학습에 사용되는 장치")
args = parser.parse_args()


def visualize_predictions(testset: Dataset, device: str, model: nn.Module, save_dir: os.PathLike, conf_thr: float = 0.1, n_images: int = 10) -> None:
    """이미지에 bbox 그려서 저장 및 시각화
    
    :param testset: 추론에 사용되는 데이터셋
    :type testset: Dataset
    :param device: 추론에 사용되는 장치
    :type device: str
    :param model: 추론에 사용되는 모델
    :type model: nn.Module
    :param save_dir: 추론한 사진이 저장되는 경로
    :type save_dir: os.PathLike
    :param conf_thr: confidence threshold - 해당 숫자에 만족하지 않는 bounding box 걸러내는 파라미터
    :type conf_thr: float
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)

    classes = ['wheat']

    model.eval()
    indices = random.choices(range(len(testset)), k=n_images)
    for i in tqdm(indices):
        image, _, image_id = testset[i]
        image = [image.to(device)]
        pred = model(image)

        image = image[0].detach().cpu().numpy().transpose(1, 2, 0)
        pred = {k: v.detach().cpu() for k, v in pred[0].items()}

        plt.imshow(image)
        ax = plt.gca()

        for box, category_id, score in zip(*pred.values()):
            if score >= conf_thr:
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                category_id = category_id.item()

                rect = patches.Rectangle(
                    (x1, y1),
                    w, h,
                    linewidth=1,
                    edgecolor='green',
                    facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(
                    x1, y1,
                    f'{classes[category_id-1]}: {score:.2f}',
                    c='white',
                    size=5,
                    path_effects=[pe.withStroke(linewidth=2, foreground='green')],
                    family='sans-serif',
                    weight='semibold',
                    va='top', ha='left',
                    bbox=dict(
                        boxstyle='round',
                        ec='green',
                        fc='green',
                    )
                )

        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f'{image_id}.jpg'), dpi=150, bbox_inches='tight', pad_inches=0)
        plt.clf()


def test(device):
    train_image_dir = 'data/global-wheat-detection/train'
    test_csv_path = 'data/global-wheat-detection/test_answer.csv'

    num_classes = 1

    test_data = WheatDataset(
        image_dir=train_image_dir,
        csv_path=test_csv_path,
        transform=transforms.ToTensor(),
    )

    model = fasterrcnn_resnet50_fpn(num_classes=num_classes+1)
    model.load_state_dict(torch.load('wheat-faster-rcnn.pth'))
    model.to(device)

    visualize_predictions(test_data, device, model, 'examples/global-wheat-detection/faster-rcnn')
    print('Saved in ./examples/global-wheat-detection/faster-rcnn')


if __name__ == '__main__':
    test(args.device)
