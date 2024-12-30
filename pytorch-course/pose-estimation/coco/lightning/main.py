import os
import argparse
import cv2
import torch.optim as optim
from torch import nn
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from src.dataset import PoseEstimationDataModule
from src.model import create_model
from src.utils import ObjectKeypointSimilarity, visualize_pose_estimation
import torch
import numpy as np
import torch.nn.functional as F


SEED = 36
L.seed_everything(SEED)


class PoseEstimationModel(L.LightningModule):
    def __init__(self, model, batch_size: int = 32):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.validation_step_outputs = []
        self.metric = None  

    def setup(self, stage=None):
        if stage == "fit" or stage == "validate":
            self.metric = ObjectKeypointSimilarity(
                image_dir=os.path.join(self.trainer.datamodule.data_path, 'val2017/val2017/'),
                ann_file=os.path.join(self.trainer.datamodule.data_path, 'person_keypoints_val2017.json')
            )

    def forward(self, x):
        outputs = self.model(x)
        return {
            "image": x,
            "output": outputs[0]['keypoints']
        }


    def training_step(self, batch, batch_idx):
        images, targets, _ = batch
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
      
        for k, v in loss_dict.items():
            self.log(f'train_{k}', v)
        self.log('train_loss', loss)
        
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets, image_ids = batch
        
        outputs = self.model(images)
        
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
        self.log("val_loss", loss, batch_size=self.batch_size)
        
        return loss

    def on_validation_epoch_start(self):
        self.validation_step_outputs = []

    def on_validation_epoch_end(self):
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

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    def predict_step(self, batch, batch_idx):
        images, _, image_ids = batch
        return self.forward(images)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=3, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }


def main(args):
    model = PoseEstimationModel(create_model(args.model), batch_size=args.batch)
    if args.device == 'gpu':
        if len(args.gpus) == 1:
            args.gpus = [int(args.gpus)]
        else:
            args.gpus = list(map(int, args.gpus.split(',')))
    elif args.device == 'cpu':
        args.gpus = 'auto'
        args.precision = 32
    
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        dirpath=args.save,
        filename=f"{args.model}-{{epoch:02d}}-{{val_loss:.2f}}",
        save_top_k=1,
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=10,
    )

    wandb_logger = WandbLogger(project="Pose Estimation")

    trainer = L.Trainer(
        accelerator=args.device,
        devices=args.gpus,
        max_epochs=args.epoch,
        precision=int(args.precision),
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping] if args.mode == "train" else [],
    )


    data_module = PoseEstimationDataModule(
        args.data,
        batch_size=args.batch,
        mode=args.mode
    )

    if args.mode == "train":
        trainer.fit(model, data_module)
        trainer.test(model, data_module)
    else:
        trainer = L.Trainer(
            accelerator=args.device,
            devices=args.gpus,
            precision=args.precision
        )
        model = PoseEstimationModel.load_from_checkpoint(args.ckpt, model=create_model(args.model))
        predictions = trainer.predict(model, data_module)
        batch_result = predictions[0]

        print("Output shape:", batch_result["output"].shape)
        
        visualize_pose_estimation(
            image=batch_result["image"][0],
            output=batch_result["output"][0],
            save_path='prediction_result.png'
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="vitpose")
    parser.add_argument("-b", "--batch_size", dest="batch", type=int, default=32)
    parser.add_argument("-e", "--epoch", type=int, default=50)
    parser.add_argument("-d", "--data_path", dest="data", type=str, default="./data/")
    parser.add_argument("-s", "--save_path", dest="save", type=str, default="./checkpoint/")
    parser.add_argument("-dc", "--device", type=str, default="gpu")
    parser.add_argument('-g', '--gpus', type=str, nargs='+', default='0')
    parser.add_argument("-p", "--precision", type=int, default=32)
    parser.add_argument("-mo", "--mode", type=str, default="train")
    parser.add_argument("-c", "--ckpt_path", dest="ckpt", type=str, default="./checkpoint/maskedrcnn-epoch=00-val_loss=0.00.ckpt")
    args = parser.parse_args()

    main(args)
