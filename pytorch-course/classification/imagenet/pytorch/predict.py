import os
import argparse

import pandas as pd
from PIL import Image
import cv2 as cv
import torch

from src.dataset import get_transform
from src.model import create_model
from src.utils import set_seed


def predict(args):
    device = args.device
    image_path = args.image_path
    model_name = args.model_name

    image = Image.open(os.path.join(image_path)).convert('RGB')
    transform = get_transform(state='predict', image_size=256)
    pred_image = transform(image).unsqueeze(0).to(device)

    model = create_model(model=model_name).to(device)
    model.load_state_dict(torch.load(f'best_epoch-imagenet-{model_name}.pth'))

    model.eval()
    with torch.no_grad():
        pred = model(pred_image)
        pred_cls = pred[0].argmax(0)

    txt_path = '../dataset/folder_num_class_map.txt'
    classes_map = pd.read_table(txt_path, header=None, sep=' ')
    classes_map.columns = ['folder', 'number', 'classes']
    
    pred_label = classes_map['classes'][pred_cls.item()]
    cv_image = cv.imread(image_path)
    cv_image = cv.resize(cv_image, (800, 600))
    cv.putText(
        cv_image,
        f'Predicted class: "{pred_cls[0]}", Predicted label: "{pred_label}"',
        (50, 50),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2
    )

    cv.imshow('Predicted Image', cv_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    set_seed(36)

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", help="학습에 사용되는 장치")
    parser.add_argument("--model", dest="model_name", default="efficientnet", help="학습에 사용되는 모델")
    parser.add_argument("--image_path", type=str, help="예측할 이미지 선택")
    args = parser.parse_args()
    
    predict(args)
