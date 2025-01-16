import os
import argparse
import json

import cv2
import numpy as np
from PIL import Image

import torch

from src.model import create_model
from src.utils import set_seed, get_transform, SEED


def predict(detection_model, data, ckpt, device):
    with open("./dataset/instances_val2017.json", 'r') as f:
        coco_data = json.load(f)
        categories = coco_data['categories']

    num_classes = 91
    model = create_model(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(ckpt))

    original_image = Image.open(data).convert("RGB")
    original_size = original_image.size
    pred_transform = get_transform(subject='pred')
    inputs = pred_transform(original_image).unsqueeze(0).to(device)


    model.eval()
    with torch.no_grad():
        outputs = model(inputs)

    if len(outputs) > 0:
        pred_boxes = outputs[0]['boxes']
        pred_labels = outputs[0]['labels']
        pred_scores = outputs[0]['scores']
        
        score_threshold = 0.8
        
        high_conf_idx = pred_scores > score_threshold
        boxes = pred_boxes[high_conf_idx].cpu().numpy()
        labels = pred_labels[high_conf_idx].cpu().numpy()
        scores = pred_scores[high_conf_idx].cpu().numpy()

        scale_x = original_size[0] / 256
        scale_y = original_size[1] / 256
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        
        
    img = cv2.imread(data)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for box, label, score in zip(boxes, labels, scores):
        color = np.random.randint(0, 255, 3).tolist()
        
        
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
    
        for category in categories:
            if category['id'] == label:
                pred_cate = category['name']
        label_text = f'{pred_cate}: {score:.2f}'
        cv2.putText(
            img,
            label_text,
            (int(box[0]), int(box[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow('Object detection results', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    set_seed(SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='faster_rcnn')
    parser.add_argument('-d', '--data_path', dest='data', type=str, default='../dataset/instance_example.jpg')
    parser.add_argument('-c', '--checkpoint_path', dest='ckpt', type=str, default='./checkpoint/best-object-detection-faster_rcnn.pth')
    parser.add_argument('-dc', '--device', type=str, default='cuda')
    args = parser.parse_args()
    
    predict(args.model, args.data, args.ckpt, args.device)
