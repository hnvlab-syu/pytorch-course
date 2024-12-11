from typing import Tuple, Dict, Any

import wandb
import torch
import numpy as np

from PIL import Image
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


class SRModel(LightningModule):
    def __init__(
            self,
            net,   # net: 실제 네트워크 구조, SRMoel: training, validation, testing 로직을 포함한 전체 모듈
            optimizer: torch.optim.Optimizer,   
            scheduler: torch.optim.lr_scheduler,  
            compile: bool,
    ):  
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.net = net  ##############

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

        # self.val_psnr_best = MaxMetric()
        # self.test_psnr_best = MaxMetric()

        # self.val_ssim_best = MaxMetric()
        # self.test_ssim_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor: # self() == self.model(inputs)
        # 3-> 12 channels  *realesrgan일 때 필요 (입력 12채널)*
        # inputs = torch.cat([inputs, inputs, inputs, inputs], dim=1) or # x = torch.cat([x] * 4, dim=1)    
        # return self.model(inputs)
        return self.net(x)  ##############

    def on_train_start(self):
        self.train_loss.reset()
        self.val_loss.reset()
        self.test_loss.reset()

        # self.val_psnr_best.reset()
        # self.test_psnr_best.reset()
        # self.val_ssim_best.reset()
        # self.test_ssim_best.reset()

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
        # psnr = self.val_psnr.compute()
        # ssim = self.val_ssim.compute()
        # psnr_best = self.val_psnr_best(self.val_psnr)
        # ssim_best = self.val_ssim_best(self.val_ssim)
        # self.log("val/psnr_best", psnr_best, sync_dist=True, prog_bar=True)
        # self.log("val/ssim_best", ssim_best, sync_dist=True, prog_bar=True)
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

    def configure_optimizers(self) -> Dict[str, Any]:
        # optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
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
