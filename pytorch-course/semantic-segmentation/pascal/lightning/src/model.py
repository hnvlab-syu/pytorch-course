from torchvision import models

def create_model(model: str = 'deeplabv3'):
    if model == 'deeplabv3':
        return _deeplabv3_resnet50_pretrained()
    elif model == 'fcn':
        return _fcn_ResNet50_pretrained()

def _deeplabv3_resnet50_pretrained():
    return models.segmentation.deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)

def _fcn_ResNet50_pretrained():
    return models.segmentation.fcn_resnet50(weights=models.segmentation.FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
  