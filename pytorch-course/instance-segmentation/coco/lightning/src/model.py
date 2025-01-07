from torchvision import models

def create_model(model='mask_rcnn', num_classes=91):
    if model =='mask_rcnn':
        return _mask_rcnn_model(num_classes)
    elif model == 'mask_rcnnv2':
        return _mask_rcnn_v2_model(num_classes)
    
def _mask_rcnn_model(num_classes=91):
    model = models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')
    
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = models.detection.mask_rcnn.MaskRCNNPredictor(
        in_channels=in_features_mask,
        dim_reduced=256,
        num_classes=num_classes
    )

    return model

def _mask_rcnn_v2_model(num_classes=91):
    model = models.detection.maskrcnn_resnet50_fpn_v2(weights='DEFAULT')
    
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = models.detection.mask_rcnn.MaskRCNNPredictor(
        in_channels=in_features_mask,
        dim_reduced=256,
        num_classes=num_classes
    )

    return model