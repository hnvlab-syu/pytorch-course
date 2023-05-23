import os
import random
import shutil

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


def visualize_predictions(testset: Dataset, device: str, model: nn.Module, save_dir: os.PathLike, n_images: int = 10, alpha: float = 0.5) -> None:
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

    model.eval()
    indices = random.choices(range(len(testset)), k=n_images)
    for i in tqdm(indices):
        image, _, meta_data = testset[i]
        image_id, height, width = meta_data.values()

        image = image.to(device)
        pred = model(image.unsqueeze(0))['out']
        pred = torch.softmax(pred, dim=1)

        max_index = torch.argmax(pred, dim=1)
        pred_bool = torch.zeros_like(pred, dtype=torch.bool).scatter(1, max_index.unsqueeze(1), True)

        image = (image * 255.0).type(torch.uint8)
        result = draw_segmentation_masks(image.cpu(), pred_bool.cpu().squeeze(), alpha=alpha)
        result = F.resize(result, size=(height, width))
        plt.imshow(result.permute(1, 2, 0).numpy())

        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f'{image_id}.jpg'), dpi=150, bbox_inches='tight', pad_inches=0)
        plt.clf()


def test():
    image_dir = 
    label_dir = 
    test_csv_path = 

    size = (500, 500)
    num_classes = 20

    test_data = PascalVOC2012Dataset(
        image_dir=image_dir,
        label_dir=label_dir,
        csv_path=test_csv_path,
        transform=get_transform(size=size),
        mask_transform=get_mask_transform(size=size)
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = deeplabv3_resnet50(num_classes=num_classes+1)
    model.load_state_dict(torch.load('pascal-voc-2012-deeplabv3.pth'))
    model.to(device)

    visualize_predictions(test_data, device, model, 'examples/pascal-voc-2012/deeplabv3', alpha=0.8)
