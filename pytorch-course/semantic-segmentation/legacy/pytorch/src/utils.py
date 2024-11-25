from glob import glob
import os
import random
from typing import Sequence

import pandas as pd

from torchvision import transforms


# 데이터셋 분리
def split_dataset(label_dir: os.PathLike, split_rate: float = 0.2) -> None:
    """데이터셋을 비율에 맞춰 train / test로 나눕니다.
    
    :param label_dir: label 데이터셋 경로
    :type path: os.PathLike
    :param split_rate: train과 test로 데이터 나누는 비율
    :type split_rate: float
    """
    print("Dataset Split")

    root_dir = os.path.dirname(label_dir)

    image_ids = []
    for path in glob(os.path.join(label_dir, '*.png')):
        file_name = os.path.split(path)[-1]
        image_id = os.path.splitext(file_name)[0]
        image_ids.append(image_id)

    random.shuffle(image_ids)

    split_point = int(split_rate * len(image_ids))
    train_ids = image_ids[split_point:]
    test_ids = image_ids[:split_point]

    train_df = pd.DataFrame({'image_id': train_ids})
    train_df.to_csv(os.path.join(root_dir, 'train_answer.csv'), index=False)

    test_df = pd.DataFrame({'image_id': test_ids})
    test_df.to_csv(os.path.join(root_dir, 'test_answer.csv'), index=False)

    print("Image len:", len(image_ids))
    print("Train len:", len(train_df), "/ Test len:", len(test_df))
    print("Done\n")

# transform 구현
def get_transform(size: Sequence[int]):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size),
    ])

# mask_transform 구현
def get_mask_transform(size: Sequence[int]):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(
            size,
            interpolation=transforms.InterpolationMode.NEAREST
        ),
    ])