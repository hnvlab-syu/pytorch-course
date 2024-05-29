from ast import literal_eval
import os
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import pandas as pd

import torch
from torch import Tensor
from torch.utils.data import Dataset


class WheatDataset(Dataset):
    """Wheat 데이터셋 사용자 정의 클래스를 정의합니다.
    """
    def __init__(
        self,
        image_dir: os.PathLike,
        csv_path: os.PathLike,
        transform: Optional[Sequence[Callable]]
        ) -> None:
        super().__init__()
        """데이터 정보를 불러와 정답(bbox)과 각각 데이터의 이름(image_id)를 저장
        
        :param image_dir: 데이터셋 경로
        :type image_dir: os.PathLike
        :param csv_path: 데이터셋 정보를 담고있는 csv 파일 경로
        :type csv_path: os.PathLike
        :param transform: 데이터셋을 정규화하거나 텐서로 변환, augmentation등의 전처리하기 위해 사용할 여러 함수들의 sequence
        :type transform: Optional[Sequence[Callable]]
        :param is_test: 테스트 데이터인지 아닌지 확인
        :type is_test: bool
        """

        df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        grouped = df.groupby(by='image_id')
        self.grouped_dict = {image_id: group for image_id, group in grouped}
        self.image_ids = tuple(self.grouped_dict.keys())
        self.transform = transform

    def __len__(self) -> int:
        """데이터셋의 길이를 반환
        
        :return: 데이터셋 길이
        :rtype: int
        """
        return len(self.image_ids)
    
    def __getitem__(self, index: int) -> Tuple:
        """데이터의 인덱스를 주면 이미지와 정답을 같이 반환하는 함수
        
        :param index: 이미지 인덱스
        :type index: int
        :return: 이미지 한장과 정답 {bbox, labels}를 같이 반환
        :rtype: Tuple[Tensor]
        """
        image_id = self.image_ids[index]
        image = Image.open(os.path.join(self.image_dir, f'{image_id}.jpg')).convert('RGB')
        boxes = np.array([literal_eval(box) for box in self.grouped_dict[image_id]['bbox']])
        boxes[: ,2] = boxes[:, 0] + boxes[:, 2]
        boxes[: ,3] = boxes[:, 1] + boxes[:, 3]
        labels = [1] * len(boxes)

        if self.transform is not None:
            image = self.transform(image)
            target = {
                'boxes': torch.as_tensor(boxes, dtype=torch.float32),
                'labels': torch.as_tensor(labels, dtype=torch.int64)
            }
        
        return image, target, image_id
    

def collate_fn(batch: Tensor) -> Tuple:
    return tuple(zip(*batch))
