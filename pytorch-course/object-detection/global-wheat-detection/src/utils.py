from ast import literal_eval
from collections import defaultdict
import json
import os

import pandas as pd
from pycocotools import COCO
from pycocotools.cocoeval import COCOeval


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

    grouped = df.groupby(by='image_id')
    grouped_list = [group for _, group in grouped]

    split_point = int(split_rate * len(grouped))

    test_ids = grouped_list[:split_point]
    train_ids = grouped_list[split_point:]

    test_df = pd.concat(test_ids)
    test_df.to_csv(os.path.join(root_dir, 'test_answer.csv'), index=False)
    train_df = pd.concat(train_ids)
    train_df.to_csv(os.path.join(root_dir, 'train_answer.csv'), index=False)


class MeanAveragePrecision:
    def __init__(self, csv_path: os.PathLike) -> None:
        self.id_csv2coco = {}
        json_path = self.to_coco(csv_path)
        self.coco_gt = COCO(json_path)

        self.detections = []

    def to_coco(self, csv_path: os.PathLike) -> os.PathLike:
        df = pd.read_csv(csv_path)
        
        grouped = df.groupby(by='image_id')
        grouped_dict = {image_id: group for image_id, group in grouped}
        
        res = defaultdict(list)

        n_id = 0
        for image_id, (file_name, group) in enumerate(grouped_dict.items()):
            res['images'].append({
                'id': image_id,
                'width': 1024,
                'height': 1024,
                'file_name': f'{file_name}.jpg',
            })

            self.id_csv2coco[file_name] = image_id

            for _, row in group.iterrows():
                x1, y1, w, h = literal_eval(row['bbox'])
                res['annotations'].append({
                    'id': n_id,
                    'image_id': image_id,
                    'category_id': 1,
                    'area': w * h,
                    'bbox': [x1, y1, w, h],
                    'iscrowd': 0,
                })
                n_id += 1

        res['categories'].extend([{'id': 1, 'name': 'wheat'}])
            
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

            p['scores'] = p['scores'].cpu().numpy()
            p['labels'] = p['labels'].cpu().numpy()

            image_id = self.id_csv2coco[image_id]
            for b, l, s in zip(*p.values()):
                self.detections.append({
                    'image_id': image_id,
                    'category_id': l,
                    'bbox': b.tolist(),
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
