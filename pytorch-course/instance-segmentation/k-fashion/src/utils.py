from collections import defaultdict
import json
import os

import numpy as np
from PIL import Image
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import albumentations as A
from albumentations.pytorch import ToTensorV2


EDGES = [
    [0, 1], [0, 2], [2, 4], [1, 3], [6, 8], [8, 10],
    [5, 7], [7, 9], [5, 11], [11, 13], [13, 15], [6, 12],
    [12, 14], [14, 16], [5, 6], [0, 17], [5, 17], [6, 17],
    [11, 12]
]


def split_dataset(csv_path: os.PathLike, split_rate: float = 0.2) -> None:
    """Dirty-MNIST 데이터셋을 비율에 맞춰 train / test로 나눕니다.
    
    :param path: Dirty-MNIST 데이터셋 경로
    :type path: os.PathLike
    :param split_rate: train과 test로 데이터 나누는 비율
    :type split_rate: float
    """
    root_dir = os.path.dirname(csv_path)

    df = pd.read_csv(csv_path)
    df = df.sample(frac=1).reset_index(drop=True)

    grouped = df.groupby(by='image')
    grouped_list = [group for _, group in grouped]

    split_point = int(split_rate * len(grouped))

    test_ids = grouped_list[:split_point]
    train_ids = grouped_list[split_point:]

    test_df = pd.concat(test_ids)
    test_df.to_csv(os.path.join(root_dir, 'test_answer.csv'), index=False)
    train_df = pd.concat(train_ids)
    train_df.to_csv(os.path.join(root_dir, 'train_answer.csv'), index=False)


def get_transform():
    return A.Compose(
        [
            ToTensorV2()
        ],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
        keypoint_params=A.KeypointParams(format='xy')
    )


class ObjectKeypointSimilarity:
    def __init__(self, image_dir: os.PathLike, csv_path: os.PathLike) -> None:
        self.image_dir = image_dir
        self.id_csv2coco = {}
        json_path = self.to_coco(csv_path)
        self.coco_gt = COCO(json_path)

        self.detections = []

    def to_coco(self, csv_path: os.PathLike) -> os.PathLike:
        df = pd.read_csv(csv_path)
    
        grouped = df.groupby(by='image')
        grouped_dict = {image_id: group for image_id, group in grouped}
        
        res = defaultdict(list)

        n_id = 0
        for image_id, (file_name, group) in enumerate(grouped_dict.items()):
            with Image.open(os.path.join(self.image_dir, file_name), 'r') as image:
                width, height = image.size
            res['images'].append({
                'id': image_id,
                'width': width,
                'height': height,
                'file_name': file_name,
            })

            self.id_csv2coco[file_name] = image_id

            for _, row in group.iterrows():
                keypoints = row[1:].values.reshape(-1, 2)

                x1 = np.min(keypoints[:, 0])
                y1 = np.min(keypoints[:, 1])
                x2 = np.max(keypoints[:, 0])
                y2 = np.max(keypoints[:, 1])

                w = x2 - x1
                h = y2 - y1

                keypoints = np.concatenate([keypoints, np.ones((24, 1), dtype=np.int64)+1], axis=1).reshape(-1).tolist()
                res['annotations'].append({
                    'keypoints': keypoints,
                    'num_keypoints': 24,
                    'id': n_id,
                    'image_id': image_id,
                    'category_id': 1,
                    'area': w * h,
                    'bbox': [x1, y1, w, h],
                    'iscrowd': 0,
                })
                n_id += 1

        res['categories'].extend([
            {
                'id': 1,
                'name': 'person',
                'keypoints': df.keys()[1:].tolist(),
                'skeleton': EDGES,
            }
        ])
            
        root_dir = os.path.split(csv_path)[0]
        save_path = os.path.join(root_dir, 'coco_annotations.json')
        with open(save_path, 'w') as f:
            json.dump(res, f)

        return save_path
    
    def update(self, preds, image_ids):
        for p, image_id in zip(preds, image_ids):
            p['boxes'][:, 2] = p['boxes'][:, 2] - p['boxes'][:, 0]
            p['boxes'][:, 3] = p['boxes'][:, 3] - p['boxes'][:, 1]
            p['boxes'] = p['boxes'].cpu().numpy()

            num_keypoints = len(p['keypoints'])
            p['keypoints'][:, :, 2] += 1
            p['keypoints'] = p['keypoints'].reshape(num_keypoints, -1) if num_keypoints > 0 else p['keypoints'] 
            p['keypoints'] = p['keypoints'].cpu().numpy()

            p['scores'] = p['scores'].cpu().numpy()
            p['labels'] = p['labels'].cpu().numpy()

            image_id = self.id_csv2coco[image_id]
            for b, l, s, k in zip(*p.values()):
                self.detections.append({
                    'image_id': image_id,
                    'category_id': l,
                    'bbox': b.tolist(),
                    'keypoints': k.tolist(),
                    'score': s
                })

    def reset(self):
        self.detections = []

    def compute(self):
        coco_dt = self.coco_gt.loadRes(self.detections)
        
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_eval = COCOeval(self.coco_gt, coco_dt, 'keypoints')
        coco_eval.params.kpt_oks_sigmas = np.ones((24, 1)) * 0.05
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
