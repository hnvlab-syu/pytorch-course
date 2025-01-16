import os
import argparse

import numpy as np
import cv2

import torch
from torch import nn
from torchmetrics import MeanMetric, MaxMetric
from torchmetrics.functional import jaccard_index
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from src.dataset import PascalVOC2012DataModule
from src.model import create_model
from src.utils import visualize_batch, SEED

L.seed_everything(SEED)


class SegmentationModel(L.LightningModule):
    def __init__(self, model, num_classes: int, batch_size: int = 32):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.train_loss = MeanMetric()
        self.train_miou = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_miou = MeanMetric()
        self.val_miou_best = MaxMetric()
        self.test_loss = MeanMetric()
        self.test_miou = MeanMetric()       
        self.losses = []

    def forward(self, inputs):
        return self.model(inputs)['out']

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs)
        loss = self.loss_fn(output, target)

        predictions = torch.argmax(output, dim=1)

        self.train_loss(loss)
        self.log("train_loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)

        miou = jaccard_index(predictions, target, num_classes=self.num_classes, task="multiclass")
        self.train_miou(miou)
        self.log("train_miou", self.train_miou, on_step=True, on_epoch=True, prog_bar=True)

        # visualize_batch(inputs, target, predictions)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs)
        loss = self.loss_fn(output, target)
        
        predictions = torch.argmax(output, dim=1)

        self.val_loss(loss)
        self.log("val_loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)

        miou = jaccard_index(predictions, target, num_classes=self.num_classes, task="multiclass")
        self.val_miou(miou)
        self.log("val_miou", self.val_miou, on_step=True, on_epoch=True, prog_bar=True)

        # visualize_batch(inputs, target, predictions)
        
        return loss

    def on_validation_epoch_end(self):
        best_miou = self.val_miou.compute()

        self.val_miou_best(best_miou)
        self.log("val_miou_best", self.val_miou_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs)
        loss = self.loss_fn(output, target)

        predictions = torch.argmax(output, dim=1)

        self.test_loss(loss)
        self.log("test_loss", self.test_loss, on_step=True, on_epoch=True, prog_bar=True)

        miou = jaccard_index(predictions, target, num_classes=self.num_classes, task="multiclass")
        self.test_miou(miou)
        self.log("test_miou", self.test_miou, on_step=True, on_epoch=True, prog_bar=True)
        
        # visualize_batch(inputs, target, predictions)

        return loss

    def on_test_epoch_end(self):
        avg_loss = self.test_loss.compute()
        avg_miou = self.test_miou.compute()
        
        self.log("test_miou_final", avg_miou, prog_bar=True)
        self.log("test_loss_final", avg_loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9)


def main(segmentation_model, data, batch, epoch, save_path, device, gpus, precision, mode, ckpt, num_workers):
    num_classes = 21

    model = SegmentationModel(
        model=create_model(segmentation_model),
        num_classes=num_classes,
        batch_size=batch
    )

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if device == 'gpu':
        if not isinstance(gpus, list):
            gpus = [int(gpus)]
        else:
            gpus = list(map(int, gpus))
    elif device == 'cpu':
        gpus = 'auto'
        precision = 32
        
    if mode == 'train':
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            dirpath=f'{save_path}',
            filename=f'{segmentation_model}-'+'{epoch:02d}-{val_loss:.2f}',
            save_top_k=1,
        )
        early_stopping = EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=10
        )
        wandb_logger = WandbLogger(project="pascal-semanticsegmentation")

        trainer = L.Trainer(
            accelerator=device,
            devices=gpus,
            max_epochs=epoch,
            precision=precision,
            logger=wandb_logger,
            callbacks=[checkpoint_callback, early_stopping],
        )
        trainer.fit(model, PascalVOC2012DataModule(data, batch, 'train', num_workers=num_workers))
        trainer.test(model, PascalVOC2012DataModule(data, batch, 'train', num_workers=num_workers))

    elif mode == 'predict':
        trainer = L.Trainer(
            accelerator=device,
            devices=gpus,
            precision=precision
        )
        
        model = SegmentationModel.load_from_checkpoint(
            checkpoint_path=ckpt,
            model=create_model(segmentation_model),
            num_classes=num_classes
        )

        predictions = trainer.predict(model, PascalVOC2012DataModule(data, 1, 'predict', num_workers=num_workers))

        for pred in predictions:
            pred_mask = pred.squeeze().cpu().numpy()
            img = cv2.imread(data)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            pred_mask = np.argmax(pred_mask, axis=0)
            pred_mask = cv2.resize(pred_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            pred_mask = (pred_mask / pred_mask.max() * 255).astype(np.uint8)
            color_mask = cv2.applyColorMap(pred_mask, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img, 0.6, color_mask, 0.4, 0)

            cv2.imshow('Predicted Segmentation', overlay)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--num_workers', type=int, default=0, help='number of worker processes for data loading')
    parser.add_argument('-m', '--model', type=str, default='deeplabv3')
    parser.add_argument('-b', '--batch_size', dest='batch', type=int, default=32)
    parser.add_argument('-e', '--epoch', type=int, default=10)
    parser.add_argument('-d', '--data_path', dest='data', type=str, default='../dataset/VOC2012')
    parser.add_argument('-s', '--save_path', dest='save', type=str, default='./checkpoint/')
    parser.add_argument('-dc', '--device', type=str, default='gpu')
    parser.add_argument('-g', '--gpus', type=str, nargs='+', default='0')
    parser.add_argument('-p', '--precision', type=int, default=32)
    parser.add_argument('-mo', '--mode', type=str, default='train')
    parser.add_argument('-c', '--ckpt_path', dest='ckpt', type=str, default='./checkpoint/')
    args = parser.parse_args()
    
    main(args.model, args.data, args.batch, args.epoch, args.save, args.device, args.gpus, args.precision, args.mode, args.ckpt, args.num_workers)
