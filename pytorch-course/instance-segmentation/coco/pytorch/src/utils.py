import random

import numpy as np
from sklearn.model_selection import train_test_split

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

def split_dataset(data_list):
    train_list, val_list = train_test_split(data_list, test_size=0.2, random_state=SEED)
    val_list, test_list = train_test_split(val_list, test_size=0.5, random_state=SEED)
    return train_list, val_list, test_list


def get_transform(subject='train'):
    if subject == 'train':
        return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    elif subject == 'test':
        return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    elif subject == 'pred':
        return transforms.Compose([
        transforms.ToTensor()
    ])

def collate_fn(batch):
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    images = torch.stack(images, dim=0)
    
    return images, targets