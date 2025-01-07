import os
import json

import cv2
import numpy as np

SEED = 36


def visualize_prediction(data, predictions):
    with open(os.path.join('../dataset/instances_val2017.json'), 'r') as f:
        coco_data = json.load(f)
        categories = coco_data['categories']

    pred_boxes = predictions['boxes']
    pred_labels = predictions['labels']
    pred_scores = predictions['scores']
    pred_masks = predictions['masks']
    
    score_threshold = 0.7
    mask_threshold = 0.7

    high_conf_idx = pred_scores > score_threshold
    boxes = pred_boxes[high_conf_idx]
    labels = pred_labels[high_conf_idx]
    scores = pred_scores[high_conf_idx]
    masks = pred_masks[high_conf_idx] > mask_threshold

    img = cv2.imread(data)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    for box, label, score, mask in zip(boxes, labels, scores, masks):
        color = np.random.randint(0, 255, 3).tolist()
        img_mask = img.copy()
        img_mask[mask] = img_mask[mask] * 0.5 + np.array(color) * 0.5
        
        for category in categories:
            if category['id'] == label:
                pred_cate = category['name']
        label_text = f'Class {pred_cate}: {score:.2f}'
        cv2.putText(
            img_mask,
            label_text,
            (int(box[0]), int(box[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )
        img = img_mask

    cv2.imshow('Instance Segmentation Result', img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
