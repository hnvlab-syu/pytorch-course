from torchvision.models.detection import keypointrcnn_resnet50_fpn


def create_model(model: str = 'resnet'):
    if model == 'maskedrcnn':
        return _masked_rcnn_pretrained()
    
def _masked_rcnn_pretrained():
    return keypointrcnn_resnet50_fpn(pretrained=True, num_classes=2, num_keypoints=17)




