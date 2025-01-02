from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric, MaxMetric
from torchmetrics.functional import jaccard_index

from src.utils import visualize_batch


class SegmentationModel(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        num_classes: int = 21,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.net = net
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.train_loss = MeanMetric()
        self.train_miou = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_miou = MeanMetric()
        self.val_miou_best = MaxMetric()
        self.test_loss = MeanMetric()
        self.test_miou = MeanMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)["out"]

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = self.loss_fn(outputs, targets)
        preds = torch.argmax(outputs, dim=1)
        return loss, preds, targets

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, preds, targets = self.model_step(batch)

        # visualize_batch(batch[0], batch[1], preds)

        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)

        train_miou = jaccard_index(preds, targets, num_classes=self.hparams.num_classes, task="multiclass")
        self.train_miou(train_miou)
        self.log("train/miou", self.train_miou, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def on_training_epoch_end(self) -> None:
        self.train_miou.reset()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)

        # visualize_batch(batch[0], batch[1], preds)

        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)

        val_miou = jaccard_index(preds, targets, num_classes=self.hparams.num_classes, task="multiclass")
        self.val_miou(val_miou)
        self.log("val/miou", self.val_miou, on_step=True, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        current_miou = self.val_miou.compute()
        self.val_miou_best(current_miou)
        self.log("val/miou_best", self.val_miou_best.compute(), sync_dist=True, prog_bar=True)

        self.val_miou.reset()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)

        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=True, on_epoch=True, prog_bar=True)

        test_miou = jaccard_index(preds, targets, num_classes=self.hparams.num_classes, task="multiclass")
        self.test_miou(test_miou)
        self.log("test/miou", self.test_miou, on_step=True, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        avg_loss = self.test_loss.compute()
        avg_miou = self.test_miou.compute()

        self.log("test/miou_final", avg_miou, prog_bar=True)
        self.log("test/loss_final", avg_loss, prog_bar=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
    
if __name__ == "__main__":
    _ = SegmentationModel(None, None, None, None)
