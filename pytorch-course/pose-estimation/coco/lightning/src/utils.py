import torch
import os
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt


class ObjectKeypointSimilarity:
    def __init__(self, image_dir: os.PathLike, ann_file: os.PathLike) -> None:
        self.coco_gt = COCO(ann_file)
        self.detections = []

    def update(self, preds, image_ids):
        for pred in preds:
            
            bbox = np.array(pred['bbox'])
            if len(bbox.shape) == 1:
                bbox = bbox.reshape(1, -1)
            
           
            keypoints = np.array(pred['keypoints']).reshape(-1).tolist()
            
            self.detections.append({
                'image_id': pred['image_id'],
                'category_id': pred['category_id'],
                'bbox': bbox.flatten().tolist(),
                'keypoints': keypoints,
                'score': pred['score']
            })

    def reset(self):
        self.detections = []

    def compute(self):
        if len(self.detections) == 0:
            print("No detections to evaluate")
            return
            
        coco_dt = self.coco_gt.loadRes(self.detections)
        
        
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'keypoints')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()


def visualize_pose_estimation(image, output, save_path):
    keypoints = output.cpu().numpy()
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy().transpose(1, 2, 0)
        image = ((image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255).astype(np.uint8)
    
    
    edges = [
        (0, 1), (0, 2), (1, 3), (2, 4), 
        (5, 6), (5, 7), (6, 8),         
        (7, 9), (8, 10),                  
        (5, 11), (6, 12),                 
        (11, 13), (12, 14),              
        (13, 15), (14, 16)             
    ]
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0.5:  
            plt.scatter(x, y, c='r', s=50)
    
    for (i, j) in edges:
        if keypoints[i][2] > 0.5 and keypoints[j][2] > 0.5:
            plt.plot([keypoints[i][0], keypoints[j][0]], 
                    [keypoints[i][1], keypoints[j][1]], 'g-')
    
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()