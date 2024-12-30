import os
import argparse

import cv2
import numpy as np
from PIL import Image

import torch

from src.model import create_model
from src.utils import set_seed, get_transform, SEED


def main(model_name, data_dir, device, ckpt):
    model = create_model(model=model_name).to(device)
    model.load_state_dict(torch.load(ckpt))

    image_dir = os.path.join(data_dir)
    inputs = Image.open(image_dir).convert("RGB")
    pred_transform = get_transform(subject='pred')
    inputs = pred_transform(inputs).unsqueeze(0).to(device)

    # predict
    model.eval()
    with torch.no_grad():
        output = model(inputs)['out']
        pred = torch.argmax(output, dim=1)

    origin_image = inputs[0].cpu().permute(1, 2, 0).numpy()

    pred_segment = pred[0].cpu().numpy()
    colors = np.random.randint(0, 255, (21, 3), dtype=np.uint8).astype(np.float32) / 255.0
    pred_colored = np.zeros((*pred_segment.shape, 3), dtype=np.float32)
    for class_id in range(21):
        pred_colored[pred_segment == class_id] = colors[class_id]

    overlay = cv2.addWeighted(origin_image, 0.6, pred_colored, 0.4, 0)

    cv2.imshow('Predicted Segmentation', overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print('Done!')


if __name__ == '__main__':
    set_seed(SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='deeplabv3')
    parser.add_argument('-d', '--data_path', dest='data', type=str, default='../dataset/example.jpg')
    parser.add_argument('-dc', '--device', type=str, default='cuda')
    parser.add_argument('-c', '--ckpt_path', dest='ckpt', type=str, default='./checkpoint/')
    args = parser.parse_args()
    
    main(args.model, args.data, args.device, args.ckpt)