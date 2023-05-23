import os
import random
import shutil

import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.utils import draw_keypoints

from src.dataset import DaconKeypointDataset
from src.utils import EDGES, get_transform


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

    model.eval()
    indices = random.choices(range(len(testset)), k=n_images)
    for i in tqdm(indices):
        image, _, image_id = testset[i]
        
        image = [image.to(device)]
        pred = model(image)
        pred = {k: v.detach().cpu() for k, v in pred[0].items() if pred[0]['scores'] >= conf_thr}

        image = (image * 255.0).type(torch.uint8)
        result = draw_keypoints(image.cpu(), pred['keypoints'], connectivity=EDGES, colors='blue', radius=4, width=3)
        plt.imshow(result.permute(1, 2, 0).numpy())

        plt.axis('off')
        plt.savefig(os.path.join(save_dir, image_id), dpi=150, bbox_inches='tight', pad_inches=0)
        plt.clf()


def test():
    image_dir = 
    test_csv_path = 
    num_classes = 1

    test_data = DaconKeypointDataset(
        image_dir=image_dir,
        csv_path=test_csv_path,
        transform=get_transform(),
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = keypointrcnn_resnet50_fpn(num_classes=num_classes+1, num_keypoints=24)
    model.load_state_dict(torch.load('dacon-keypoint-rcnn.pth'))
    model.to(device)

    visualize_predictions(test_data, device, model, 'examples/dacon-keypoint/keypoint-rcnn')