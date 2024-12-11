import os
import wandb
import torch
import argparse
import cv2 as cv2
import numpy as np
import lightning as L
import subprocess

from PIL import Image
from typing import Tuple

from torchmetrics import MeanMetric
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar, DeviceStatsMonitor

from src.dataset import DIV2KDataModule
from src.model import create_model


SEED = 36
L.seed_everything(SEED)

class SRModel(L.LightningModule):
    def __init__(
            self,
            model,   # model(net): 실제 네트워크 구조, SRMoel: training, validation, testing 로직을 포함한 전체 모듈
            # optimizer: torch.optim.Optimizer,   
            # scheduler: torch.optim.lr_scheduler,  
            # compile: bool,
            learning_rate: None
    ):  
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.model = model  ##############
        self.learning_rate = learning_rate
        self.loss_fn = torch.nn.L1Loss()    # self.criterion

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_psnr = PeakSignalNoiseRatio()
        self.val_psnr = PeakSignalNoiseRatio()
        self.test_psnr = PeakSignalNoiseRatio()

        self.train_ssim = StructuralSimilarityIndexMeasure()
        self.val_ssim = StructuralSimilarityIndexMeasure()
        self.test_ssim = StructuralSimilarityIndexMeasure()

    def forward(self, x: torch.Tensor) -> torch.Tensor: # self() == self.model(inputs)
        # 3-> 12 channels  *realesrgan일 때 필요 (입력 12채널)*
        # inputs = torch.cat([inputs, inputs, inputs, inputs], dim=1) or # x = torch.cat([x] * 4, dim=1)    
        # return self.model(inputs)
        return self.model(x)  ##############

    def on_train_start(self):
        self.train_loss.reset()
        self.val_loss.reset()
        self.test_loss.reset()

    def model_step(
            self, 
            batch: Tuple[torch.Tensor, torch.Tensor]  # (저해상도 이미지, 고해상도 이미지)
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lr_imgs, hr_imgs = batch

        sr_imgs = self.forward(lr_imgs)  # sr: 생성된 고해상도
        loss = self.loss_fn(sr_imgs, hr_imgs)  # L1Loss로 복원 품질 측정

        # # 이미지 품질 메트릭 계산 (PSNR, SSIM)
        # psnr = self.psnr(sr_imgs, hr_imgs)  
        # ssim = self.ssim(sr_imgs, hr_imgs)

        return loss, lr_imgs, sr_imgs, hr_imgs    

    def training_step(self, batch): # 여기서의 batch는 DataLoader가 제공하는 실제 데이터 배치: (lr_imgs, hr_imgs)형태
        loss, _, sr_imgs, hr_imgs = self.model_step(batch)

        self.train_loss.update(loss)
        psnr = self.train_psnr(sr_imgs, hr_imgs)
        ssim = self.train_ssim(sr_imgs, hr_imgs)
        self.log("train_loss", self.train_loss, on_step=True, prog_bar=True)
        self.log("train_psnr", psnr, on_step=True, prog_bar=True)
        self.log("train_ssim", ssim, on_step=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        psnr = self.train_psnr.compute()
        ssim = self.train_ssim.compute()

        self.log("train_psnr_epoch", psnr, sync_dist=True, on_epoch=True, prog_bar=True)
        self.log("train_ssim_epoch", ssim, sync_dist=True, on_epoch=True, prog_bar=True)
        
        self.train_psnr.reset()
        self.train_ssim.reset()

    def validation_step(self, batch, batch_idx):
        loss, lr_imgs, sr_imgs, hr_imgs = self.model_step(batch)

        self.val_loss.update(loss)
        psnr = self.val_psnr(sr_imgs, hr_imgs)
        ssim = self.val_ssim(sr_imgs, hr_imgs)
        self.log("val_loss", self.val_loss, on_step=True, prog_bar=True)
        self.log("val_psnr", psnr, on_step=True, prog_bar=True)
        self.log("val_ssim", ssim, on_step=True, prog_bar=True)

        if batch_idx == 0 and self.logger:
            self._log_images(lr_imgs[0], sr_imgs[0], hr_imgs[0])

    def on_validation_epoch_end(self):
        psnr = self.val_psnr.compute()
        ssim = self.val_ssim.compute()

        self.log("val_psnr_epoch", psnr, sync_dist=True, on_epoch=True, prog_bar=True)
        self.log("val_ssim_epoch", ssim, sync_dist=True, on_epoch=True, prog_bar=True)
        
        self.val_psnr.reset()
        self.val_ssim.reset()

    def test_step(self, batch, batch_idx):
        loss, _, sr_imgs, hr_imgs = self.model_step(batch)

        self.test_loss.update(loss)
        psnr = self.test_psnr(sr_imgs, hr_imgs)
        ssim = self.test_ssim(sr_imgs, hr_imgs)
        self.log("test_loss", self.test_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test_psnr", psnr, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test_ssim", ssim, on_step=True, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        # """Lightning hook that is called when a test epoch ends."""
        # acc = self.test_loss.compute()
        # self.test_acc_best(acc)
        # self.log("test/acc_best", self.test_acc_best.compute(), sync_dist=True, prog_bar=True)
        psnr = self.test_psnr.compute()
        ssim = self.test_ssim.compute()

        self.log("test_psnr_epoch", self.test_psnr, sync_dist=True, prog_bar=True)
        self.log("test_ssim_epoch", self.test_ssim, sync_dist=True, prog_bar=True)
        
        self.test_psnr.reset()
        self.test_ssim.reset()

    def predict_step(self, batch, batch_idx):
        # lr_imgs, _ = batch
        print('----------model.py > predict_step------------', batch.shape)     # (b,C,H,W)
        lr_imgs = batch
        sr_imgs = self(lr_imgs)   # forward
        return sr_imgs
    
    def _log_images(self, lr_img, sr_img, hr_img):
        if self.logger:
            # 0-1 범위에서 0-255 범위로 변환 후 uint8로 변환
            lr_np = (lr_img.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            sr_np = (sr_img.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            hr_np = (hr_img.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
            # PIL Image로 변환
            lr_pil = Image.fromarray(lr_np)
            sr_pil = Image.fromarray(sr_np)
            hr_pil = Image.fromarray(hr_np)
    
            # 개별적으로 로깅 (사이즈가 다르기 때문에 리스트로 그룹화해서 로깅하면 에러)
            self.logger.experiment.log({
                "LR_image": wandb.Image(lr_pil, caption="LR (120x120)"),
                "SR_image": wandb.Image(sr_pil, caption="SR (480x480)"),
                "HR_image": wandb.Image(hr_pil, caption="HR (480x480)")
            })
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  
                "interval": "epoch",    
                "frequency": 1,
        }
    }


def main(model, weights, upscale, lr_data, hr_data, learning_rate, batch_size, num_workers, pin_memory, epoch, save, device, gpus, precision, mode, ckpt):
    if not os.path.exists(save):
        os.makedirs(save)

    if device == 'gpu':
        if len(gpus) == 1:
            gpus = [int(gpus)]
        else:
            gpus = list(map(int, gpus.split(',')))
    elif device == 'cpu':
        gpus = 'auto'
        precision = 32

    if mode in ['predict']:
        batch_size = 1

    datamodule = DIV2KDataModule(
        lr_dir = lr_data,
        hr_dir = hr_data,
        batch_size = batch_size,
        num_workers = num_workers,
        pin_memory = pin_memory,
        mode = mode,
        upscale = upscale,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=save,
            filename=f'sr_model-'+'{epoch:02d}-{val_ssim:.2f}',
            monitor='val_ssim',
            mode='max',
            save_top_k=3,
        ),
        EarlyStopping(
            monitor='val_ssim',
            mode='max',
            patience=5
        ),
        RichProgressBar(),
        DeviceStatsMonitor()
    ]

    
    if mode == 'train':
        trainer = L.Trainer(
            accelerator = device,
            devices = gpus,   # if device == 'gpu' else None,
            max_epochs = epoch,
            precision = precision,
            logger = WandbLogger(project="SR",),
            callbacks = callbacks,
        )
        sr_model = SRModel(create_model(model, weights, upscale), learning_rate)
        trainer.fit(sr_model, datamodule)
        trainer.test(sr_model, datamodule)

    # elif mode == 'test':
    #     sr_model = SRModel.load_from_checkpoint(ckpt, model=create_model(model, weights, upscale))
    #     test_output = trainer.test(sr_model, datamodule)

    #     save_dir = os.path.join(save, 'test_output')
    #     os.makedirs(save_dir, exist_ok=True)
        
    #     # with open(os.path.join(save_dir, 'metrics.txt'), 'w') as f:
    #     #     for metric, value in test_output[0].items():
    #     #         f.write(f"{metric}: {value}\n")
    #     for i, output in enumerate(test_output):
    #         print('---------------------shape---------------------')
    #         print(output.shape)     
    #         img_np = output.cpu().numpy().squeeze().transpose(1, 2, 0)   
    #         print(img_np)
    #         img_np = (img_np * 255).clip(0, 255).astype(np.uint8)  
    #         print(img_np)
    
    #         im = Image.fromarray(img_np)  
    #         im.save(os.path.join(save_dir, f'output.png'))

    else:  # predict/inferece
        trainer = L.Trainer(
            accelerator=device,
            devices=gpus,
            precision=precision
        )
        sr_model = SRModel.load_from_checkpoint(ckpt, model=create_model(model, weights, upscale))
        predict_output = trainer.predict(sr_model, datamodule)
        
        save_dir = os.path.join(save, 'predict_output')
        os.makedirs(save_dir, exist_ok=True)
        
        for i, output in enumerate(predict_output):
            print('---------------------shape---------------------')
            print(output.shape)     
            img_np = output.cpu().numpy().squeeze().transpose(1, 2, 0)   
            print(img_np)
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)  
            print(img_np)
    
            im = Image.fromarray(img_np)  
            im.save(os.path.join(save_dir, f'output.png'))


if __name__ == "__main__":
    subprocess.run("python download_pretrained_models.py EDSR", shell=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='edsr')
    parser.add_argument('-w', '--weights', type=str, default='./weights/EDSR/EDSR_Mx4_f64b16_DIV2K_official-0c287733.pth')
    parser.add_argument('-u', '--upscale', type=int, default=4)
    parser.add_argument('-lrd', '--lr_data_path', dest='lr_data', type=str, default='../datasets/DIV2K_train_LR_mild_sub')
    parser.add_argument('-hrd', '--hr_data_path', dest='hr_data', type=str, default='../datasets/DIV2K_train_HR_sub')     # '/DIV2K/dataset/DIV2K_valid_HR'
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-b', '--batch_size', type=int, default=512)       # 1/ dest='batch'
    parser.add_argument('-n', '--num_workers', type=int, default=4)       # 0
    parser.add_argument('-pin', '--pin_memory', type=bool, default=True)       
    parser.add_argument('-e', '--epoch', type=int, default=150)
    parser.add_argument('-s', '--save_path', dest='save', type=str, default='./checkpoint/')
    parser.add_argument('-dc', '--device', type=str, default='gpu')
    parser.add_argument('-g', '--gpus', type=str, nargs='+', default='0')
    parser.add_argument('-p', '--precision', type=str, default='16-mixed')  # 32-true/ 16-mixed
    parser.add_argument('-mo', '--mode', type=str, default='train')
    parser.add_argument('-c', '--ckpt_path', dest='ckpt', type=str, default='./checkpoint/sr_model-epoch=03-val_ssim=0.52.ckpt')
    args = parser.parse_args()

    main(args.model, args.weights, args.upscale, args.lr_data, args.hr_data, args.learning_rate, args.batch_size, args.num_workers, args.pin_memory,
         args.epoch, args.save, args.device, args.gpus, args.precision, args.mode, args.ckpt)  

