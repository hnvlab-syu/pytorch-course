import random

import numpy as np
import cv2

import torch
from torchvision import transforms

SEED = 36

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def visualize_batch(inputs, target, predictions):
    image, mask, pred = inputs[0], target[0], predictions[0]

    image_to_show = image.cpu().permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
    image_to_show = cv2.cvtColor(image_to_show, cv2.COLOR_RGB2BGR)  # RGB -> BGR 변환

    mask_to_show = mask.cpu().numpy()
    pred_to_show = pred.cpu().numpy()
    colors = np.random.randint(0, 255, (21, 3), dtype=np.uint8).astype(np.float32) / 255.0
    mask_colored = np.zeros((*mask_to_show.shape, 3), dtype=np.float32)  # (H, W, 3)
    for class_id in range(21):
        mask_colored[mask_to_show == class_id] = colors[class_id]
    pred_colored = np.zeros((*pred_to_show.shape, 3), dtype=np.float32)  # (H, W, 3)
    for class_id in range(21):
        pred_colored[pred_to_show == class_id] = colors[class_id]

    combined = np.hstack((image_to_show, mask_colored, pred_colored))
    cv2.imshow('Image and Mask', combined)
    cv2.waitKey(0)

def preprocess_mask(mask):
    mask = np.array(mask)
    mask[mask == 255] = 0   
    return torch.tensor(mask, dtype=torch.long)

def get_transform(subject):
    if subject == 'image':
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    elif subject == 'mask':
        return transforms.Compose([
            transforms.Resize((256, 256))
        ])
    elif subject == 'pred':
        return transforms.Compose([
            transforms.ToTensor()
        ])

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images)  
    targets = torch.stack(targets)    
    return images, targets