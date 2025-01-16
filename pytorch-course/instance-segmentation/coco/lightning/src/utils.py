import os
import json
import random

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

def visualize_batch(input_image, target, output):
    input_image = input_image.cpu().numpy().transpose(1, 2, 0)
    input_image = (input_image * 255).astype(np.uint8)

    target_overlay = input_image.copy()
    pred_overlay = input_image.copy()
    colormaps = [cv2.COLORMAP_AUTUMN,
        cv2.COLORMAP_BONE,
        cv2.COLORMAP_JET,
        cv2.COLORMAP_WINTER,
        cv2.COLORMAP_RAINBOW,
        cv2.COLORMAP_OCEAN,
        cv2.COLORMAP_SUMMER,
        cv2.COLORMAP_SPRING,
        cv2.COLORMAP_COOL,
        cv2.COLORMAP_HSV,
        cv2.COLORMAP_PINK,
        cv2.COLORMAP_HOT,
        cv2.COLORMAP_PARULA,
        cv2.COLORMAP_MAGMA,
        cv2.COLORMAP_INFERNO,
        cv2.COLORMAP_PLASMA,
        cv2.COLORMAP_VIRIDIS
    ]
    
    if len(target['masks']) > 0:        
        for idx, mask in enumerate(target['masks']):
            target_mask = mask.cpu().numpy()
            target_mask = (target_mask * 255).astype(np.uint8)
            colormap = colormaps[idx % len(colormaps)]
            target_mask_colored = cv2.applyColorMap(target_mask, colormap)
            target_overlay = cv2.addWeighted(target_overlay, 0.5, target_mask_colored, 0.5, 0)
    else:
        target_overlay = input_image.copy()

    if len(output['masks']) > 0:        
        for i, pred_mask in enumerate(output['masks']):
            pred_mask = pred_mask.detach().cpu().numpy()
            pred_mask = ((pred_mask[0] > 0.7) * 255).astype(np.uint8)
            colormap = colormaps[i % len(colormaps)]
            pred_mask_colored = cv2.applyColorMap(pred_mask, colormap)
            pred_overlay = cv2.addWeighted(pred_overlay, 0.5, pred_mask_colored, 0.5, 0)
    else:
        pred_overlay = input_image.copy()

    combined_image = np.hstack((input_image, target_overlay, pred_overlay))

    cv2.imshow("Target (Left) / Prediction (Right)", combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()