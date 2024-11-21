from typing import Any, Dict, List, Tuple

import cv2
import pandas as pd
import hydra
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from src.models import SegmentationModel
from src.models.components.semantic_segmentation_model import create_model
import numpy as np
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
       
    predictions = trainer.predict(
        model=model,
        datamodule=datamodule,ckpt_path=cfg.ckpt_path
    )
    data = './2007_000027.jpg'  
    
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



@hydra.main(version_base="1.3", config_path="../configs", config_name="predict.yaml")
def main(cfg: DictConfig) -> None:

    predict(cfg)


if __name__ == "__main__":
    main()