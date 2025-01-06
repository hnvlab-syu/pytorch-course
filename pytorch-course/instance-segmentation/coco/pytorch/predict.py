import os
import argparse
import json

import cv2
import numpy as np
from PIL import Image

import torch

from src.model import create_model
from src.utils import set_seed, get_transform, SEED


def predict(segmentation_model, data, ckpt, device):
    with open("../dataset/instances_val2017.json", 'r') as f:
        coco_data = json.load(f)
        categories = coco_data['categories']

    num_classes = 91
    model = create_model(segmentation_model, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(ckpt))

    image_dir = os.path.join(data)
    inputs = Image.open(image_dir).convert("RGB")
    pred_transform = get_transform(subject='pred')
    inputs = pred_transform(inputs).unsqueeze(0).to(device)

    # pred
    model.eval()
    with torch.no_grad():
        inputs = [image.to(device) for image in inputs]

        outputs = model(inputs)

    if len(outputs) > 0:
        pred_boxes = outputs[0]['boxes']
        pred_labels = outputs[0]['labels']
        pred_scores = outputs[0]['scores']
        pred_masks = outputs[0]['masks']
        
        score_threshold = 0.7
        mask_threshold = 0.7
        
        high_conf_idx = pred_scores > score_threshold
        boxes = pred_boxes[high_conf_idx].cpu().numpy()
        labels = pred_labels[high_conf_idx].cpu().numpy()
        scores = pred_scores[high_conf_idx].cpu().numpy()
        masks = pred_masks[high_conf_idx] > mask_threshold
        masks = masks.squeeze(1).cpu().numpy()

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


if __name__ == '__main__':
    set_seed(SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='mask_rcnn')
    parser.add_argument('-d', '--data_path', dest='data', type=str, default='../dataset/instance_example.jpg')
    parser.add_argument('-c', '--checkpoint_path', dest='ckpt', type=str, default='./checkpoint/')
    parser.add_argument('-dc', '--device', type=str, default='cuda')
    args = parser.parse_args()
    
    predict(args.model, args.data, args.ckpt, args.device)