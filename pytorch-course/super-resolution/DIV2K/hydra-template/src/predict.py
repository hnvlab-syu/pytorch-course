from typing import List

import os
import cv2
import numpy as np
import hydra
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from PIL import Image

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
    print(object_dict)

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting predicting!")
    output = trainer.predict(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    print('------------predict.py > output---------------\n', output)

    save_dir = cfg.output_dir   # os.path.join(cfg.root_dir, 'output')
    os.makedirs(save_dir, exist_ok=True)

    for i, output in enumerate(output):
        print('---------------------shape---------------------')
        print(output.shape)     # 
        img_np = output.cpu().numpy().squeeze().transpose(1, 2, 0)   # squeeze(차원 따로 설정 안 했을 경우, 1인 차원 전부 제거)
        print(img_np)
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)   ##########
        print(img_np)

        im = Image.fromarray(img_np)   ########## cv2말고 PIL로
        im.save(os.path.join(save_dir, f'output.png'))


@hydra.main(version_base="1.3", config_path="../configs", config_name="predict.yaml")
def main(cfg: DictConfig) -> None:
    predict(cfg)


if __name__ == "__main__":
    main()