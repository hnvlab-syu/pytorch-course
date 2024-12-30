import os
import json
import warnings
import argparse

import torch.optim as optim
from torchmetrics.detection import MeanAveragePrecision

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.dataset import COCODataModule
from src.model import create_model

warnings.filterwarnings('ignore')


class TrainingModule(pl.LightningModule):
    def __init__(
            self,
            model,
    ):
        super().__init__()
        self.model = model
        # self.batch_size = args.batch_size
        self.metric_fn = MeanAveragePrecision()

    def forward(self, images, targets):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        self.log('loss', losses, on_step=True, on_epoch=False, prog_bar=True)
        self.log('class_loss', loss_dict['loss_classifier'], on_step=True, on_epoch=False, prog_bar=True)
        self.log('box_loss', loss_dict['loss_box_reg'], on_step=True, on_epoch=False, prog_bar=True)
        self.log('obj_loss', loss_dict['loss_objectness'], on_step=True, on_epoch=False, prog_bar=True)

        return losses

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images)
        metric.update(outputs, targets)
        return outputs
    
    def validation_epoch_end(self, outputs):
        metric_compute = metric.compute()
        map = metric_compute['map'].numpy().tolist()
        map_50 = metric_compute['map_50'].numpy().tolist()
        map_75 = metric_compute['map_75'].numpy().tolist()
        
        self.log('val_mAP', map, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mAP_50', map_50, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mAP_75', map_75, on_step=False, on_epoch=True, prog_bar=True)
        metric.reset()
    
    def predict_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images)
        return outputs

    def configure_optimizers(self):
        return optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0005)
        # return optim.Adam(self.model.parameters(), lr=1e-4)


def main(args):
    pl.seed_everything(42)
    
    datamodule = COCODataModule(
        datasets_path = args.datasets_path, 
        annots_path = args.annots_path, 
        batch_size = args.batch_size,
        num_workers = args.num_workers
    )

    model = TrainingModule(create_model(args.model))
    
    if args.mode == 'train':
        ckpt_path = args.weight_path
        num_gpus = args.gpus
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        
        checkpoint_callback = ModelCheckpoint(
            monitor='val_mAP',
            dirpath=ckpt_path,
            filename='{epoch}-{val_mAP:.2f}',
            save_top_k=-1,
            mode='max',
            save_weights_only=True,
        )
        early_stopping_callback = EarlyStopping(
            monitor='val_mAP',
            min_delta=0.00,
            patience=100,
            verbose=False,
            mode='min'
        )
        
        trainer = pl.Trainer(
            log_every_n_steps=1,
            logger=wandb_logger,
            max_epochs=args.max_epochs,
            accelerator="gpu",
            strategy='ddp_find_unused_parameters_false',
            gpus = num_gpus,
            precision=16,
            callbacks=[checkpoint_callback, early_stopping_callback]
            # callbacks=[checkpoint_callback]
        )

        trainer.fit(model, datamodule)
        
    if args.mode == 'test':
        model = model.load_from_checkpoint(args.checkpoint)

        trainer = pl.Trainer(
            gpus=1,
            precision=16
        )
        
        output_lists = trainer.predict(model, datamodule)
        outputs = []
        for i in range(len(output_lists)):
            outputs += output_lists[i]
        # outputs = outputs[0]+ outputs[1]
        print(len(outputs))
        
        # with open(args.valid_annt_path, "r") as f:
        #     annots = json.load(f)
        with open(os.path.join(args.annots_path, "captions_val2017"), 'r') as f:
            annots = json.load(f)

        images = annots['images']

        image_ids = []
        for image in images:
            image_id = image['id']
            image_ids.append(image_id)

        # eval
        predicts = []
        idx = 0
        for i in range(len(outputs)):
            # print(outputs[i])
            boxes = outputs[i]['boxes'].detach().cpu().numpy().tolist()
            scores = outputs[i]['scores'].detach().cpu().numpy().tolist()
            labels = outputs[i]['labels'].detach().cpu().numpy().tolist()

            for bbox,label,score in zip(boxes,labels,scores):
                # print(bbox,label,score)
                bbox[2] = bbox[2] - bbox[0]
                bbox[3] = bbox[3] - bbox[1]
                tmp = {"image_id": int(image_ids[idx]), "category_id": int(label), "bbox": bbox, "score": float(score)}
                predicts.append(tmp)
            idx += 1

        with open('predict.json', 'w') as f:
            json.dump(predicts, f)
        
        # coco_gt = COCO(args.valid_annt_path)
        coco_gt = COCO(os.path.join(args.annots_path, "captions_val2017"))
        coco_pred = coco_gt.loadRes('predict.json')
        coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Faster-RCNN')
    parser.add_argument('--image_size', type=int, default=384)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--project', type=str, default='object-detection')
    parser.add_argument('--weight_path', type=str, default='weights/faster/')
    parser.add_argument('-d', '--datasets_path', type=str, default='../datasets/')
    parser.add_argument('-a', '--annots_path', type=str, default='../datasets/annotations')
    parser.add_argument('-m', '--model', type=str, default='fastrcnn')
    parser.add_argument('-mo', '--mode', type=str, default='train')
    parser.add_argument('--gpus', type=list, default=[0,1,2,3])
    parser.add_argument('--checkpoint', type=str, default='coco_weights/faster/best.ckpt')
    args = parser.parse_args()
    
    if args.mode == 'train':
        wandb_logger = WandbLogger(project=args.project)
        metric = MeanAveragePrecision()
        
    main(args)