from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def create_model(model='faster_rcnn', num_classes=81):
    if model =='faset_rcnn':
        return faset_rcnn(num_classes)
    elif model == 'faster_rcnn_v2':
        return faster_rcnn_v2(num_classes)

def faset_rcnn(num_classes=91):
    model = models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model
    
def faster_rcnn_v2(num_classes=91):
    model = models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box = FastRCNNPredictor(in_features, num_classes)
    
    return model