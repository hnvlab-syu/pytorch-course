from typing import Tuple

import torch
import lightning as L
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class SegmentationModel(L.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        num_classes: int = 91,
    ) -> None:
        super().__init__()
        self.net = net

        self.val_map = MeanAveragePrecision()
        self.best_val_map = 0
        self.test_map = MeanAveragePrecision()

    def forward(self, inputs):
        return self.net(inputs)
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        inputs = [input.to(self.device) for input in inputs]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        outputs = self.net(inputs, targets)

        cls_loss = outputs.get('loss_classifier', 0)
        box_loss = outputs.get('loss_box_reg', 0)
        mask_loss = outputs.get('loss_mask', 0)

        loss = cls_loss + box_loss + mask_loss

        self.log('train-total_loss', loss)
        self.log('train-cls_loss', cls_loss)
        self.log('train-box_loss', box_loss)
        self.log('train-mask_loss', mask_loss)
        
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        inputs = [input.to(self.device) for input in inputs]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        outputs = self.net(inputs)
        self.val_map.update(outputs, targets)
        
        return outputs

    def on_validation_epoch_end(self):
        val_mAP = self.val_map.compute()
        self.val_map.reset()

        if self.best_val_map < val_mAP['map'].item():
            self.best_val_map = val_mAP['map'].item()

        self.log('val_mAP', val_mAP['map'].item()) 
        self.log('best_val_mAP', self.best_val_map) 

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        inputs = [input.to(self.device) for input in inputs]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets] 

        outputs = self.net(inputs, targets)
        self.test_map.update(outputs, targets)
        
        return outputs
    
    def on_test_epoch_end(self):
        test_mAP = self.test_map.compute()
        self.test_map.reset()

        self.log('test_mAP', test_mAP['map'].item())

    def predict_step(self, batch, batch_idx):
        inputs, _ = batch
        outputs = self.net(inputs)
        
        if len(outputs) > 0:
            pred_boxes = outputs[0]['boxes']
            pred_labels = outputs[0]['labels']
            pred_scores = outputs[0]['scores']
            pred_masks = outputs[0]['masks']
            
            score_threshold = 0.7
            mask_threshold = 0.7
            
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
            }
        return None
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.net.parameters(), lr=1e-3, momentum=0.9)
