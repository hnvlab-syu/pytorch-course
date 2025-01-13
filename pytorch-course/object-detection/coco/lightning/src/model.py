from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
# from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights


def _fastrrcnn_pretrained():
    return fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)   # (pretrained=True)

def _fastrrcnnv2_pretrained():
    return fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)

# def _ssdhead_pretrained():
#     return ssd300_vgg16(weights=SSD300_VGG16_Weights)

def create_model(model_type: str = 'fasterrcnn'):
    print(f"----------Creating model: {model_type}")
    if model_type.lower() == 'fasterrcnn':
        return _fastrrcnn_pretrained()
    
    elif model_type.lower() == 'fasterrcnnv2':
        return _fastrrcnnv2_pretrained()

    # elif model_type.lower() == 'ssdhead':
    #     return _ssdhead_pretrained()