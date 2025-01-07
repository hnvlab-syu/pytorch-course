import rootutils
from omegaconf import DictConfig

import hydra
from lightning import LightningDataModule, LightningModule, Trainer

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import (
    RankedLogger,
    visualize_prediction
)

log = RankedLogger(__name__, rank_zero_only=True)


def predict(cfg: DictConfig) -> None:
    assert cfg.ckpt_path, "Checkpoint path (ckpt_path) is required."
    assert cfg.pred_image, "Prediction image (pred_image) is required."

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    predictions = trainer.predict(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)[0]
    
    if predictions is not None:
        visualize_prediction(cfg.pred_image, predictions)

@hydra.main(version_base="1.3", config_path="../configs", config_name="predict.yaml")
def main(cfg: DictConfig) -> None:

    predict(cfg)


if __name__ == "__main__":
    main()
