from torchvision import models

def create_model(model: str = 'mask_rcnn', backbone: str = 'resnet50', num_classes: int = 91):
    if model =='mask_rcnn':
        return _mask_rcnn_model(backbone, num_classes)
    elif model == 'mask_rcnnv2':
        return _mask_rcnn_v2_model(backbone, num_classes)
    
def _mask_rcnn_model(backbone: str = 'resnet50', num_classes: int = 91):
    if backbone == 'resnet50':
        model = models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')
    
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model

def _mask_rcnn_v2_model(backbone: str = 'resnet50', num_classes: int = 91):
    if backbone == 'resnet50':
        model = models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')
    
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model