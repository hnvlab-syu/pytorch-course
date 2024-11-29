from typing import List

import cv2
import numpy as np
import hydra
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import (
    RankedLogger,
    instantiate_loggers,
    log_hyperparameters,
)

log = RankedLogger(__name__, rank_zero_only=True)

def predict(cfg: DictConfig) -> None:
    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting predicting!")
    outputs = trainer.predict(
        model=model, 
        datamodule=datamodule, 
        ckpt_path=cfg.ckpt_path
    )
    
    # Process each output (batch)
    for batch_idx, (lr_img, sr_img) in enumerate(outputs):
        # Convert tensors to numpy arrays
        lr_img = lr_img.cpu().numpy().transpose(1, 2, 0)
        sr_img = sr_img.cpu().numpy().transpose(1, 2, 0)
        
        # Denormalize if needed
        lr_img = np.clip(lr_img * 255, 0, 255).astype(np.uint8)
        sr_img = np.clip(sr_img * 255, 0, 255).astype(np.uint8)
        
        # Convert to BGR for OpenCV
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_RGB2BGR)
        sr_img = cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR)
        
        # Optional: Resize for display
        display_height = 400
        lr_h, lr_w = lr_img.shape[:2]
        sr_h, sr_w = sr_img.shape[:2]
        
        lr_display = cv2.resize(
            lr_img, 
            (int(lr_w * display_height/lr_h), display_height)
        )
        sr_display = cv2.resize(
            sr_img, 
            (int(sr_w * display_height/sr_h), display_height)
        )
        
        # Create comparison view
        comparison = np.hstack([lr_display, sr_display])
        
        # Add text labels
        cv2.putText(
            comparison,
            'LR Input',
            (50, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        cv2.putText(
            comparison,
            'SR Output',
            (lr_display.shape[1] + 50, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Display results
        cv2.imshow(f'Super Resolution Result {batch_idx}', comparison)
        
        # Save results
        save_path = f'output/sr_result_{batch_idx}.png'
        cv2.imwrite(save_path, comparison)
        log.info(f"Saved result to {save_path}")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


@hydra.main(version_base="1.3", config_path="../configs", config_name="predict.yaml")
def main(cfg: DictConfig) -> None:
    predict(cfg)


if __name__ == "__main__":
    main()