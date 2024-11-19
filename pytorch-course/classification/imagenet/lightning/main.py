import os
import argparse

import numpy as np
import cv2 as cv2
import pandas as pd
import torch
from torch import nn
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from src.dataset import ImageNetDataModule
from src.utils import rename_dir
from src.model import create_model


SEED = 36
L.seed_everything(SEED)
class ClassficationModel(L.LightningModule):
    def __init__(self,model, batch_size: int = 32):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.loss_fn = nn.CrossEntropyLoss()
        self.losses = []
        self.labels = []
        self.predictions = []

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.model(inputs)
        loss = self.loss_fn(output, target)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.model(inputs)
        loss =  self.loss_fn(output, target)
        _, predictions = torch.max(output, 1)

        target_np = target.detach().cpu().numpy()
        predict_np = predictions.detach().cpu().numpy()
        
        self.losses.append(loss)
        self.labels.append(np.int16(target_np))
        self.predictions.append(np.int16(predict_np))
        self.log('valid_loss', loss)
        return loss
    
    def on_validation_epoch_end(self):
        labels = np.concatenate(np.array(self.labels, dtype=object))
        predictions = np.concatenate(np.array(self.predictions, dtype=object))
        acc = sum(labels == predictions)/len(labels)

        labels = labels.tolist()
        predictions = predictions.tolist()
        loss = sum(self.losses)/len(self.losses)

        self.log('val_epoch_acc', acc)
        self.log('val_epoch_loss', loss)
        
        self.losses.clear()
        self.labels.clear()
        self.predictions.clear()
    
    def test_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.model(inputs)
        loss =  self.loss_fn(output, target)
        _, predictions = torch.max(output, 1)

        target_np = target.detach().cpu().numpy()
        predict_np = predictions.detach().cpu().numpy()
        
        self.losses.append(loss)
        self.labels.append(np.int16(target_np))
        self.predictions.append(np.int16(predict_np))
        self.log('test_loss', loss)
        return loss
    
    def on_test_epoch_end(self):
        labels = np.concatenate(np.array(self.labels, dtype = object))
        predictions = np.concatenate(np.array(self.predictions, dtype = object))
        acc = sum(labels == predictions)/len(labels)

        labels = labels.tolist()
        predictions = predictions.tolist()
        loss = sum(self.losses)/len(self.losses)

        self.log('test_epoch_acc', acc)
        self.log('test_epoch_loss', loss)
        
        self.losses.clear()
        self.labels.clear()
        self.predictions.clear()

    def predict_step(self, batch, batch_idx):
        inputs, img = batch
        output = self.model(inputs)
        _, pred_cls = torch.max(output, 1)

        return pred_cls.detach().cpu().numpy(), img

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9)
    

def main(classification_model, data, batch, epoch, save_path, device, gpus, precision, mode, ckpt):
    rename_dir()
    model = ClassficationModel(create_model(classification_model))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if device == 'gpu':
        if len(gpus) == 1:
            gpus = [int(gpus)]
        else:
            gpus = list(map(int, gpus.split(',')))
    elif device == 'cpu':
        gpus = 'auto'
        precision = 32
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_epoch_acc',
        mode='max',
        dirpath= f'{save_path}',
        filename= f'{classification_model}-'+'{epoch:02d}-{val_epoch_acc:.2f}',
        save_top_k=1,
    )
    early_stopping = EarlyStopping(
        monitor='val_epoch_acc',
        mode='max',
        patience=10
    )
    wandb_logger = WandbLogger(project="ImageNet")
    
    if mode == 'train':
        trainer = L.Trainer(
            accelerator=device,
            devices=gpus,
            max_epochs=epoch,
            precision=precision,
            logger=wandb_logger,
            callbacks=[checkpoint_callback, early_stopping],
        )
        trainer.fit(model, ImageNetDataModule(data, batch))
        trainer.test(model, ImageNetDataModule(data, batch))
    else:
        trainer = L.Trainer(
            accelerator=device,
            devices=gpus,
            precision=precision
        )
        model = ClassficationModel.load_from_checkpoint(ckpt, model=create_model(classification_model))
        pred_cls, img = trainer.predict(model, ImageNetDataModule(data, mode='predict'))[0]
        txt_path = '../dataset/folder_num_class_map.txt'
        classes_map = pd.read_table(txt_path, header=None, sep=' ')
        classes_map.columns = ['folder', 'number', 'classes']
        
        pred_label = classes_map['classes'][pred_cls.item()]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (800, 600))
        cv2.putText(
            img,
            f'Predicted class: "{pred_cls[0]}", Predicted label: "{pred_label}"',
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2
        )
        cv2.imshow('Predicted output', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='resnet')
    parser.add_argument('-b', '--batch_size', dest='batch', type=int, default=32)
    parser.add_argument('-e', '--epoch', type=int, default=50)
    parser.add_argument('-d', '--data_path', dest='data', type=str, default='../dataset')
    parser.add_argument('-s', '--save_path', dest='save', type=str, default='./checkpoint/')
    parser.add_argument('-dc', '--device', type=str, default='gpu')
    parser.add_argument('-g', '--gpus', type=str, nargs='+', default='0')
    parser.add_argument('-p', '--precision', type=str, default='32-true')
    parser.add_argument('-mo', '--mode', type=str, default='train')
    parser.add_argument('-c', '--ckpt_path', dest='ckpt', type=str, default='./checkpoint/')
    args = parser.parse_args()
    
    main(args.model, args.data, args.batch, args.epoch, args.save, args.device, args.gpus, args.precision, args.mode, args.ckpt)