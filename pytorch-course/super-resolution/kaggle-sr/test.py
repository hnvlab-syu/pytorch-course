import os
import random
import shutil

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

from src.dataset import KaggleSRDataset
from src.model import EDSR


def visualize_predictions(testset: Dataset, device: str, model: nn.Module, save_dir: os.PathLike, n_images: int = 10) -> None:
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
        lr_image, hr_image, image_id = testset[i]

        lr_image = lr_image.to(device)
        pred = model(lr_image.unsqueeze(0))

        pred = (pred.squeeze(0) * 255.0).type(torch.uint8)
        hr_image = (hr_image * 255.0).type(torch.uint8)
        
        fig, axs = plt.subplots(ncols=2, squeeze=False)
        for i, img in enumerate([hr_image, pred]):
            img = img.detach()
            img = TF.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f'{image_id}.jpg'), dpi=150, bbox_inches='tight', pad_inches=0)
        plt.clf()
        plt.close()


def test():
    lr_dir = 
    hr_dir = 
    test_csv_path = 

    test_data = KaggleSRDataset(
        lr_dir=lr_dir,
        hr_dir=hr_dir,
        csv_path=test_csv_path,
        transform=transforms.ToTensor()
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EDSR()
    model.load_state_dict(torch.load('kaggle-sr-edsr.pth'))
    model.to(device)

    visualize_predictions(test_data, device, model, 'examples/kaggle-sr/edsr')
