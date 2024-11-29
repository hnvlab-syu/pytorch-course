from typing import Tuple, Dict, Any

import torch
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
        # self.model = model
        self.save_hyperparameters(logger=False)
        # self.save_hyperparameters(logger=False)
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

        self.val_psnr_best = MaxMetric()
        self.test_psnr_best = MaxMetric()

        self.val_ssim_best = MaxMetric()
        self.test_ssim_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = torch.cat([x] * 4, dim=1)     ***realesrgan일 때 필요*** 입력이 12채널***
        # return self.model(inputs)
        return self.net(x)  ##############

    def on_train_start(self) -> None:
        self.train_loss.reset()
        self.val_loss.reset()
        self.test_loss.reset()

        self.train_psnr.reset()
        self.val_psnr.reset()
        self.test_psnr.reset()

        self.train_ssim.reset()
        self.val_ssim.reset()
        self.test_ssim.reset()

        self.val_psnr_best.reset()
        self.test_psnr_best.reset()
        self.val_ssim_best.reset()
        self.test_ssim_best.reset()

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

        return loss, sr_imgs, hr_imgs    

    def training_step(self, batch):
        loss, sr_imgs, hr_imgs = self.model_step(batch)

        self.train_loss.update(loss)
        psnr = self.train_psnr(sr_imgs, hr_imgs)
        ssim = self.train_ssim(sr_imgs, hr_imgs)
        self.log("train/loss", self.train_loss.compute(), on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/psnr", psnr, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/ssim", ssim, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(self, batch, batch_idx):
        loss, sr_imgs, hr_imgs = self.model_step(batch)

        self.val_loss.update(loss)
        psnr = self.val_psnr(sr_imgs, hr_imgs)
        ssim = self.val_ssim(sr_imgs, hr_imgs)
        self.log("val/loss", self.val_loss.compute(), on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/psnr", psnr, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/ssim", ssim, on_step=True, on_epoch=True, prog_bar=True)


    def on_validation_epoch_end(self) -> None:
        psnr = self.val_psnr.compute()
        ssim = self.val_ssim.compute()
        psnr_best = self.val_psnr_best(psnr)
        ssim_best = self.val_ssim_best(ssim)
        self.log("val/psnr_best", psnr_best, sync_dist=True, prog_bar=True)
        self.log("val/ssim_best", ssim_best, sync_dist=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, sr_imgs, hr_imgs = self.model_step(batch)

        self.test_loss.update(loss)
        psnr = self.test_psnr(sr_imgs, hr_imgs)
        ssim = self.test_ssim(sr_imgs, hr_imgs)
        self.log("test/loss", self.test_loss.compute(), on_step=True, on_epoch=True, prog_bar=True)
        self.log("test/psnr", psnr, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test/ssim", ssim, on_step=True, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        # """Lightning hook that is called when a test epoch ends."""
        # acc = self.test_loss.compute()
        # self.test_acc_best(acc)
        # self.log("test/acc_best", self.test_acc_best.compute(), sync_dist=True, prog_bar=True)
        pass

    def predict_step(self, batch):
        lr_imgs, _ = batch
        return self(lr_imgs)

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
