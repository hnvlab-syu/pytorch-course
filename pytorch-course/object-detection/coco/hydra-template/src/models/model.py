from typing import Any, Dict

import torch
import lightning.pytorch as L

from torchmetrics.detection import MeanAveragePrecision


SEED = 36
L.seed_everything(SEED)

class DetectionModel(L.LightningModule):
    def __init__(
            self,
            net: torch.nn.Module,
            optimizer: torch.optim.Optimizer,    
            scheduler: torch.optim.lr_scheduler,
            compile: bool
    ):
        super().__init__()
        print(f"Initializing DetectionModel with net: {net}")
        print(type(net))
        self.save_hyperparameters(logger=False)
        self.net = net
        self.train_mAP = MeanAveragePrecision()
        self.val_mAP = MeanAveragePrecision()
        self.test_mAP = MeanAveragePrecision()

    def forward(self, images, targets):
        return self.net(images, targets)

    def on_train_start(self):
        self.val_mAP.reset()

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.net(images, targets)   # multi-task loss(여러 개의 loss): 모델 내부에서 loss function 계산

        loss = sum(loss for loss in loss_dict.values())

        self.log('train/total_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/loss_classifier', loss_dict['loss_classifier'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/loss_box_reg', loss_dict['loss_box_reg'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/loss_objectness', loss_dict['loss_objectness'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/loss_rpn_box_reg', loss_dict['loss_rpn_box_reg'], on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.net(images)

        self.val_mAP.update(outputs, targets)
    
    def on_validation_epoch_end(self):
        metric_compute = self.val_mAP.compute()
        map = metric_compute['map'].numpy().tolist()
        map_50 = metric_compute['map_50'].numpy().tolist()
        map_75 = metric_compute['map_75'].numpy().tolist()
        
        self.log('val/mAP', map, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/mAP_50', map_50, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/mAP_75', map_75, on_step=False, on_epoch=True, prog_bar=True)
        self.val_mAP.reset()
    
    def test_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.net(images)

        self.test_mAP.update(outputs, targets)

    def on_test_epoch_end(self, outputs):
        metric_compute = self.test_mAP.compute()
        map = metric_compute['map'].numpy().tolist()
        map_50 = metric_compute['map_50'].numpy().tolist()
        map_75 = metric_compute['map_75'].numpy().tolist()
        
        self.log('test/mAP', map, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/mAP_50', map_50, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/mAP_75', map_75, on_step=False, on_epoch=True, prog_bar=True)
    
    def predict_step(self, batch, batch_idx):
        images = [batch]
        return self.net(images)  
    
    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.net.parameters())  # self.trainer.model.parameters()
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/mAP",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

if __name__ == "__main__":
    _ = DetectionModel(None, None, None, None)