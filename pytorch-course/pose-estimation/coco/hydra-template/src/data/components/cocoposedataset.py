import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import numpy as np



class CocoPoseDataset(Dataset):
    def __init__(
        self, 
        img_dir: str, 
        ann_file: str, 
        transform=None
    ):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.transform = transform
        self.image_ids = []
        for img_id in self.coco.imgs.keys():
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=[1], iscrowd=False)
            if len(ann_ids) > 0:
                self.image_ids.append(img_id)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=[1], iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)[0]
        
        
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        image = np.asarray(image)
        
        keypoints = np.array(anns['keypoints']).reshape(-1, 3)[:, :2]  
        boxes = np.array([anns['bbox']], dtype=np.float32)
        
        boxes[0, 2] += boxes[0, 0]
        boxes[0, 3] += boxes[0, 1]

        targets = {
            'image': image,
            'bboxes': boxes,
            'labels': np.array([1]),
            'keypoints': keypoints
        }

        if self.transform is not None:
            targets = self.transform(**targets)
            
            image = targets['image']
            image = image / 255.0

            targets = {
                'boxes': torch.as_tensor(targets['bboxes'], dtype=torch.float32),
                'labels': torch.as_tensor(targets['labels'], dtype=torch.int64),
                'keypoints': torch.as_tensor(
                    np.concatenate([targets['keypoints'], np.ones((17, 1))], axis=1)[np.newaxis], 
                    dtype=torch.float32
                )
            }

        return image, targets, img_info['file_name']