from typing import Any, Dict, List, Tuple

import cv2
import pandas as pd
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
    # output  = trainer.predict(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    # pred_cls, img = output[0]
    # txt_path = '../dataset/folder_num_class_map.txt'
    # classes_map = pd.read_table(txt_path, header=None, sep=' ')
    # classes_map.columns = ['folder', 'number', 'classes']
    
    # pred_label = classes_map['classes'][pred_cls.item()]
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, (800, 600))
    # cv2.putText(
    #     img,
    #     f'Predicted class: "{pred_cls[0]}", Predicted label: "{pred_label}"',
    #     (50, 50),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.8,
    #     (0, 0, 0),
    #     2
    # )
    # cv2.imshow('Predicted output', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


@hydra.main(version_base="1.3", config_path="../configs", config_name="predict.yaml")
def main(cfg: DictConfig) -> None:

    predict(cfg)


if __name__ == "__main__":
    main()
