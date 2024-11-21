import os
import argparse

import numpy as np
import cv2
import pandas as pd
import torch
from torch import nn
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from src.dataset import VOCDataModule
from src.model import create_model

SEED = 36
L.seed_everything(SEED)

class SegmentationModel(L.LightningModule):
    def __init__(self, model, num_classes: int, batch_size: int = 32):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.loss_fn = nn.CrossEntropyLoss()

        
        self.losses = []
        self.all_predictions = []
        self.all_labels = []

    def forward(self, inputs):
        return self.model(inputs)['out']

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs)
        loss = self.loss_fn(output, target)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs)
        loss = self.loss_fn(output, target)

        
        _, predictions = torch.max(output, dim=1)

        
        self.losses.append(loss)
        self.all_predictions.append(predictions.cpu())
        self.all_labels.append(target.cpu())

        self.log('valid_loss', loss)
        return loss

    def on_validation_epoch_end(self):
        
        predictions = torch.cat(self.all_predictions, dim=0)
        labels = torch.cat(self.all_labels, dim=0)

        
        ious = []
        for cls in range(self.num_classes):
            intersection = ((predictions == cls) & (labels == cls)).sum().item()
            union = ((predictions == cls) | (labels == cls)).sum().item()
            if union > 0:
                ious.append(intersection / union)

        miou = sum(ious) / len(ious) if ious else 0
        avg_loss = torch.stack(self.losses).mean()
        self.log('val_miou', miou)
        self.log('val_epoch_loss', avg_loss)

       
        self.losses.clear()
        self.all_predictions.clear()
        self.all_labels.clear()

    def test_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs)
        loss = self.loss_fn(output, target)

        _, predictions = torch.max(output, dim=1)

        ious = []
        for cls in range(self.num_classes):
            intersection = ((predictions == cls) & (target == cls)).sum().item()
            union = ((predictions == cls) | (target == cls)).sum().item()
            if union > 0:
                ious.append(intersection / union)

        miou = sum(ious) / len(ious) if ious else 0

        self.all_predictions.append(predictions.cpu())
        self.all_labels.append(target.cpu())

        self.losses.append(loss)
        self.log('test_loss', loss)
        self.log('test_miou', miou, prog_bar=True)

        return {'loss': loss, 'miou': miou}
    def on_test_epoch_end(self):
        avg_loss = torch.stack(self.losses).mean()
        self.log('test_epoch_loss', avg_loss)

        if self.all_predictions and self.all_labels:
            all_predictions = torch.cat(self.all_predictions, dim=0)
            all_labels = torch.cat(self.all_labels, dim=0)
            ious = []
            for cls in range(self.num_classes):
                intersection = ((all_predictions == cls) & (all_labels == cls)).sum().item()
                union = ((all_predictions == cls) | (all_labels == cls)).sum().item()
                if union > 0:
                    ious.append(intersection / union)

            test_miou = sum(ious) / len(ious) if ious else 0
            self.log('test_epoch_miou', test_miou)
        else:
            self.log('test_epoch_miou', 0.0)

        self.losses.clear()
        self.all_predictions.clear()
        self.all_labels.clear()

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9)



def main(segmentation_model, data, batch, epoch, save_path, device, gpus, precision, mode, ckpt):
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
            monitor='val_epoch_loss',
            mode='min',
            dirpath=f'{save_path}',
            filename=f'{segmentation_model}-'+'{epoch:02d}-{val_epoch_loss:.2f}',
            save_top_k=1,
        )
        early_stopping = EarlyStopping(
            monitor='val_epoch_loss',
            mode='min',
            patience=10
        )
        wandb_logger = WandbLogger(project="VOC_Segmentation")

        trainer = L.Trainer(
            accelerator=device,
            devices=gpus,
            max_epochs=epoch,
            precision=precision,
            logger=wandb_logger,
            callbacks=[checkpoint_callback, early_stopping],
        )
        trainer.fit(model, VOCDataModule(data, batch_size=batch))
        trainer.test(model, VOCDataModule(data, batch_size=batch))

   
    elif mode == 'predict':
        trainer = L.Trainer(
            accelerator=device,
            devices=gpus,
            precision=precision
        )

       
        model = SegmentationModel.load_from_checkpoint(
            ckpt,
            model=create_model(segmentation_model),
            num_classes=num_classes
        )

       
        predictions = trainer.predict(
            model,
            VOCDataModule(data, batch_size=1, mode='predict') 
        )

        
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
    parser.add_argument('-m', '--model', type=str, default='fcn')
    parser.add_argument('-b', '--batch_size', dest='batch', type=int, default=32)
    parser.add_argument('-e', '--epoch', type=int, default=1)
    parser.add_argument('-d', '--data_path', dest='data', type=str, default='./data/VOC2012')
    parser.add_argument('-s', '--save_path', dest='save', type=str, default='./checkpoint/')
    parser.add_argument('-dc', '--device', type=str, default='gpu')
    parser.add_argument('-g', '--gpus', type=str, nargs='+', default='0')
    parser.add_argument('-p', '--precision', type=int, default=32)
    parser.add_argument('-mo', '--mode', type=str, default='train')
    parser.add_argument('-c', '--ckpt_path', dest='ckpt', type=str, default='./checkpoint/')
    args = parser.parse_args()
    
    main(args.model, args.data, args.batch, args.epoch, args.save, args.device, args.gpus, args.precision, args.mode, args.ckpt)
