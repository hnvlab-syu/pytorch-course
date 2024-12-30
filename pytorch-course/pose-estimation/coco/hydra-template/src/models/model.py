from typing import Any, Dict, Tuple
import os
import torch
from torch import nn
from lightning import LightningModule
from src.utils.utils import ObjectKeypointSimilarity

class PoseEstimationModel(LightningModule):
    def __init__(
        self,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool = False,
        num_classes: int = 1,
        batch_size: int = 32,
        heatmap_size: int = 64,
        sigma: int = 2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.net = net
        self.validation_step_outputs = []
        self.metric = None
        

    def setup(self, stage=None):
        if stage == "fit" or stage == "validate":
            self.metric = ObjectKeypointSimilarity(
                image_dir=os.path.join(self.trainer.datamodule.data_path, 'val2017/val2017/'),
                ann_file=os.path.join(self.trainer.datamodule.data_path, 'person_keypoints_val2017.json')
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.net(x)
        return {
            "image": x,
            "output": outputs[0]['keypoints']
        }

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, Any], batch_idx: int) -> torch.Tensor:
        images, targets, _ = batch
        loss_dict = self.net(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        
        for k, v in loss_dict.items():
            self.log(f'train_{k}', v)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, Any], batch_idx: int) -> torch.Tensor:
        images, targets, image_ids = batch
        outputs = self.net(images)
        
        results = []
        img_ids = []
        
        for idx, output in enumerate(outputs):
            keypoints = output['keypoints'].detach().cpu()
            scores = output['scores'].detach().cpu()
            boxes = output['boxes'].detach().cpu()
            
            if len(scores) > 0:
                max_score_idx = scores.argmax()
                result = {
                    'image_id': int(image_ids[idx].split('.')[0]),
                    'category_id': 1,
                    'keypoints': keypoints[max_score_idx].flatten().tolist(),
                    'score': scores[max_score_idx].item(),
                    'bbox': boxes[max_score_idx].tolist()
                }
                results.append(result)
                img_ids.append(int(image_ids[idx].split('.')[0]))
        
        self.validation_step_outputs.append({
            'results': results,
            'image_ids': img_ids
        })
        
        loss_dict = {}
        for k, v in outputs[0].items():
            if 'loss' in k:
                loss_dict[k] = v
        
        loss = sum(loss for loss in loss_dict.values())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def on_validation_epoch_start(self) -> None:
        self.validation_step_outputs = []

    def on_validation_epoch_end(self) -> None:
        all_results = []
        all_image_ids = []
        
        for output in self.validation_step_outputs:
            all_results.extend(output['results'])
            all_image_ids.extend(output['image_ids'])
        
        if all_results:
            self.metric.update(all_results, all_image_ids)
            results = self.metric.compute()
            
            if results is not None:
                for k, v in results.items():
                    self.log(f"val_{k}", v)
        
        self.validation_step_outputs = []

    def predict_step(
    self, 
        batch: Tuple[torch.Tensor, torch.Tensor, Any], 
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        images, _, image_ids = batch
        return self.forward(images)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.net.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

if __name__ == "__main__":
    _ = PoseEstimationModel(None, None, None)
