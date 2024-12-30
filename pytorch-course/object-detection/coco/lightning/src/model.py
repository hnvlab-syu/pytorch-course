from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
# from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights


def _fastrcnn_pretrained():
    return fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights)   # (pretrained=True)

def _ssdhead_pretrained():
    return ssd300_vgg16(weights=SSD300_VGG16_Weights)

# def _efficientb0_pretrained():
#     return efficientnet_b0(weights=EfficientNet_B0_Weights)


def create_model(model_type: str = 'fastrcnn'):
    if model_type.lower == 'fastrcnn':
        return _fastrcnn_pretrained()
    
    elif model_type.lower == 'ssdhead':
        return _ssdhead_pretrained()
    
    # elif model_type.lower == 'efficientnet':
    #     return _efficientb0_pretrained()

