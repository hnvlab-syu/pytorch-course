import csv
import os
from typing import Callable, Sequence, Tuple

import numpy as np
from PIL import Image

from torch import Tensor
from torch.utils.data import Dataset


class DirtyMnistDataset(Dataset):
    """Dirty-MNIST 데이터셋 사용자 정의 클래스를 정의합니다.
    """
    def __init__(
        self,
        image_dir: os.PathLike,
        csv_path: os.PathLike,
        transform: Sequence[Callable]
    ) -> None:
        """데이터 정보를 불러와 정답(label)과 각각 데이터의 이름(image_id)를 저장
        
        :param dir: 데이터셋 경로
        :type dir: os.PathLike
        :param image_ids: 데이터셋의 정보가 담겨있는 csv 파일 경로
        :type image_ids: os.PathLike
        :param transforms: 데이터셋을 정규화하거나 텐서로 변환, augmentation등의 전처리하기 위해 사용할 여러 함수들의 sequence
        :type transforms: Sequence[Callable]
        """
        super().__init__()

        self.image_dir = image_dir
        self.transform = transform

        self.labels = {}
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.labels[int(row[0])] = list(map(int, row[1:]))

        self.image_ids = list(self.labels.keys())

    def __len__(self) -> int:
        """데이터셋의 길이를 반환
        
        :return: 데이터셋 길이
        :rtype: int
        """
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Tuple[Tensor]:
        """데이터의 인덱스를 주면 이미지와 정답을 같이 반환하는 함수
        
        :param index: 이미지 인덱스
        :type index: int
        :return: 이미지 한장과 정답 값
        :rtype: Tuple[Tensor]
        """
        image_id = self.image_ids[index]
        image = Image.open(os.path.join(self.image_dir, f'{str(image_id).zfill(5)}.png')).convert('RGB')
        target = np.array(self.labels.get(image_id))

        if self.transform is not None:
            image = self.transform(image)

        return image, target
