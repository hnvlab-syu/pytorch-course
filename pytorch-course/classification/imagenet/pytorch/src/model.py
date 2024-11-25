from torch import nn, Tensor
from torchvision.models import (
    vgg16, VGG16_Weights,
    resnet18, ResNet18_Weights,
    efficientnet_b0, EfficientNet_B0_Weights
)


def create_model(model: str):
    if model == 'vgg':
        return _vgg16_pretrained()
    elif model == 'resnet':
        return _resnet18_pretrained()
    elif model == 'efficientnet':
        return _efficientb0_pretrained()

def _vgg16_pretrained():
    return vgg16(weights=(VGG16_Weights.IMAGENET1K_V1))

def _resnet18_pretrained():
    return resnet18(weights=(ResNet18_Weights.IMAGENET1K_V1))

def _efficientb0_pretrained():
    return efficientnet_b0(weights=(EfficientNet_B0_Weights.IMAGENET1K_V1))
