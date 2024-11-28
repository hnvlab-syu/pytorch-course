import torch
import wandb
import lightning as L
import numpy as np
from PIL import Image

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.archs.edsr_arch import EDSR

from torch import nn
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure


def create_model(model_name: str, pretrained_path: str | tuple):   # pretrained_path: args.weights
    if model_name.lower() == 'esrgan':
        model = RRDBNet(
            num_in_ch=3,  # RGB
            num_out_ch=3, # RGB
        )
        if pretrained_path:  # 'weights/ESRGAN.pth'
            weights = torch.load(pretrained_path, weights_only=True)    # .pth 파일 안의 가중치 데이터만 로드, 실행 가능한 객체 무시 (unpickle의 잠재적인 보안 문제를 예방)
                
    elif model_name.lower() == 'realesrgan':
        model = RRDBNet(
            num_in_ch=12,   # 3->12
            num_out_ch=3,
            scale=4,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
        )
        if pretrained_path:
            weights = torch.load(pretrained_path, weights_only=True)

    elif model_name.lower() == 'edsr':
        model = EDSR(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=16,
                upscale=4,
                res_scale=1,
                img_range=255.,
                rgb_mean=(0.4488, 0.4371, 0.4040)
            )
        if pretrained_path:
            weights = torch.load(pretrained_path, weights_only=True)

    return model


class SRModel(L.LightningModule):
    def __init__(self, model, learning_rate, batch_size):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.loss_fn = nn.L1Loss()
        self.psnr = PeakSignalNoiseRatio().to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure().to(self.device)
        
        self.save_hyperparameters(ignore=['model'])

    def forward(self, inputs):      # self() == self.model(inputs)
        # 3-> 12 channels ***realesrgan일 때 필요***
        # inputs = torch.cat([inputs, inputs, inputs, inputs], dim=1)
        return self.model(inputs)
    
    def training_step(self, batch, ):  # 여기서의 batch는 DataLoader가 제공하는 실제 데이터 배치: (lr_imgs, hr_imgs)형태
        lr_imgs, hr_imgs = batch
        sr_imgs = self(lr_imgs)

        loss = self.loss_fn(sr_imgs, hr_imgs)
        psnr = self.psnr(sr_imgs, hr_imgs)
        ssim = self.ssim(sr_imgs, hr_imgs)
        
        self.log_dict({
            'train_loss': loss,
            'train_psnr': psnr,
            'train_ssim': ssim,
            'epoch': self.current_epoch
        }, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        lr_imgs, hr_imgs = batch
        sr_imgs = self(lr_imgs)
        
        loss = self.loss_fn(sr_imgs, hr_imgs)
        psnr = self.psnr(sr_imgs, hr_imgs)
        ssim = self.ssim(sr_imgs, hr_imgs)
        
        self.log_dict({
            'val_loss': loss,
            'val_psnr': psnr,
            'val_ssim': ssim,
            'epoch': self.current_epoch
        }, prog_bar=True)
        
        if batch_idx == 0 and self.logger:
            self._log_images(lr_imgs[0], sr_imgs[0], hr_imgs[0])
            
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def predict_step(self, batch, batch_idx):
        lr_imgs, _ = batch
        return self(lr_imgs)
    
    
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
    
            # 개별적으로 로깅
            self.logger.experiment.log({
                "LR_image": wandb.Image(lr_pil, caption="LR (120x120)"),
                "SR_image": wandb.Image(sr_pil, caption="SR (480x480)"),
                "HR_image": wandb.Image(hr_pil, caption="HR (480x480)")
            })

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_psnr"
            }
        }
        



















