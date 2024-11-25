import os
import random
import shutil
import warnings

import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.utils import draw_segmentation_masks

from src.dataset import PascalVOC2012Dataset
from src.utils import get_transform, get_mask_transform

# 경고 메세지 무시
warnings.filterwarnings("ignore")

def visualize_predictions(testset: Dataset, device: str, model: nn.Module, save_dir: os.PathLike, n_images: int = 10, alpha: float = 0.7) -> None:
    # 디렉토리 재생성
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)

    # 랜덤 이미지 예측
    model.eval()
    indices = random.choices(range(len(testset)), k=n_images)

    for i in tqdm(indices):
        image, _, meta_data = testset[i]
        image_id, height, width = meta_data.values()

        # 이미지 예측
        image = image.to(device)
        pred = model(image.unsqueeze(0))['out']
        pred = torch.softmax(pred, dim=1)

        # 세그멘테이션 맵 생성
        max_index = torch.argmax(pred, dim=1)
        pred_bool = torch.zeros_like(pred, dtype=torch.bool).scatter(1, max_index.unsqueeze(1), True)

        # 이미지 mask 그리고 시각화
        image = (image * 255.0).type(torch.uint8)
        result = draw_segmentation_masks(image.cpu(), pred_bool.cpu().squeeze(), alpha=alpha) # mask 시각화
        result = F.interpolate(result.unsqueeze(0), size=(height, width), mode='nearest').squeeze(0) # 이미지 resize
        plt.imshow(result.permute(1, 2, 0).numpy())

        # 저장 및 초기화
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f"{image_id}.jpg"), dpi=150, bbox_inches='tight', pad_inches=0)
        plt.clf()


def test():
    # 디렉토리 설정
    image_dir = 'data/VOC2012/JPEGImages'
    label_dir = 'data/VOC2012/SegmentationClass'
    test_csv_path = 'data/VOC2012/test_answer.csv'

    # 하이퍼파라미터 초기화
    size = (500, 500)
    num_classes = 20

    # 데이터셋 초기화
    test_data = PascalVOC2012Dataset(
        image_dir=image_dir,
        label_dir=label_dir,
        csv_path=test_csv_path,
        num_classes=num_classes,
        transform=get_transform(size=size),
        mask_transform=get_mask_transform(size=size)
    )

    # device 초기화
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 모델 초기화
    model = deeplabv3_resnet50(pretrained=True, num_classes=num_classes+1)
    model.load_state_dict(torch.load('pascal-voc-2012-pretrained-deeplabv3.pth'))
    model.to(device)

    # 객체 예측 및 시각화
    print(f"\n<<<   Test by {device}   >>>\n")
    visualize_predictions(test_data, device, model, 'examples/deeplabv3', alpha=0.7)


if __name__ == "__main__":
    test()