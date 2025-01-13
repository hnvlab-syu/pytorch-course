from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights


def _fastrrcnn_pretrained():
    return fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)   # (pretrained=True)

def _fastrrcnnv2_pretrained():
    return fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)

def create_model(model: str = 'fasterrcnn'):
    print(f"----------Creating model: {model}")
    if model.lower() == 'fasterrcnn':
        return _fastrrcnn_pretrained()
    
    elif model.lower() == 'fasterrcnnv2':
        return _fastrrcnnv2_pretrained()


if __name__ == '__main__':
    model = create_model('fasterrcnnv2')
    print(model.parameters())
