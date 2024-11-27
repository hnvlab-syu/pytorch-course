import torch
import lightning as L
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def create_model(model_name: str):
    if model_name == "mask_rcnn":
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        
        # ROI heads 수정
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=91)
        
        # Mask predictor 수정
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            num_classes=91
        )
        return model
    else:
        raise ValueError(f"Unknown model: {model_name}")

class SegmentationModel(L.LightningModule):
    def __init__(
        self,
        net: dict,
        batch_size: int = 32,
    ):
        super().__init__()
        
        self.save_hyperparameters(logger=False)
        self.model = net
        self.batch_size = batch_size
            
        self.train_map = MeanAveragePrecision()
        self.val_map = MeanAveragePrecision()
        self.test_map = MeanAveragePrecision()
        
    def forward(self, inputs):
        return self.model(inputs)
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        inputs = [input.to(self.device) for input in inputs]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        outputs = self.model(inputs, targets)

        classification_loss = outputs.get('loss_classifier', 0)
        box_loss = outputs.get('loss_box_reg', 0)
        mask_loss = outputs.get('loss_mask', 0)

        total_loss = classification_loss + box_loss + mask_loss

        self.log('train_loss', total_loss)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        inputs = [input.to(self.device) for input in inputs]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        outputs = self.model(inputs)

        self.val_map.update(outputs, targets)
        
        return outputs
    
    def on_validation_epoch_end(self):
        val_mAP = self.val_map.compute()
        self.log('val_mAP', val_mAP['map']) 
        self.val_map.reset()

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        inputs = [input.to(self.device) for input in inputs]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets] 

        outputs = self.model(inputs, targets)
            
        self.test_map.update(outputs, targets)
        
        return outputs
    
    def on_test_epoch_end(self):
        test_mAP = self.test_map.compute()
        self.log('test_mAP', test_mAP['map'])  
        self.test_map.reset()

    def predict_step(self, batch, batch_idx):
        inputs, img = batch
        outputs = self.model(inputs)
        
        if len(outputs) > 0:
            pred_boxes = outputs[0]['boxes']
            pred_labels = outputs[0]['labels']
            pred_scores = outputs[0]['scores']
            pred_masks = outputs[0]['masks']
            
            score_threshold = 0.5
            mask_threshold = 0.5
            
            high_conf_idx = pred_scores > score_threshold
            boxes = pred_boxes[high_conf_idx].cpu().numpy()
            labels = pred_labels[high_conf_idx].cpu().numpy()
            scores = pred_scores[high_conf_idx].cpu().numpy()
            masks = pred_masks[high_conf_idx] > mask_threshold
            masks = masks.squeeze(1).cpu().numpy()
            
            return {
                'boxes': boxes,
                'labels': labels,
                'scores': scores,
                'masks': masks
            }, img
        
        return None, img
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)