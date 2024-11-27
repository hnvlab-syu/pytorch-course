from typing import Any, Dict, List, Tuple

import cv2
import pandas as pd
import hydra
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
<<<<<<< HEAD
import numpy as np
=======
>>>>>>> upstream/develop

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

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    predictions, img = trainer.predict(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)[0]

    if predictions is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        
        for box, label, score, mask in zip(
            predictions['boxes'], 
            predictions['labels'], 
            predictions['scores'],
            predictions['masks']
        ):
            color = np.random.randint(0, 255, 3).tolist()
            img_mask = img.copy()
            img_mask[mask] = img_mask[mask] * 0.5 + np.array(color) * 0.5
            
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_mask, (x1, y1), (x2, y2), color, 2)
        
            label_text = f'Class {label}: {score:.2f}'
            cv2.putText(
                img_mask,
                label_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
            img = img_mask

        cv2.imshow('Instance Segmentation Result', img)
        key = cv2.waitKey(0) 
        if key == ord('q'):  # q를 누르면 종료
            cv2.destroyAllWindows()
            return

                


@hydra.main(version_base="1.3", config_path="../configs", config_name="predict.yaml")
def main(cfg: DictConfig) -> None:

    predict(cfg)


if __name__ == "__main__":
    main()
