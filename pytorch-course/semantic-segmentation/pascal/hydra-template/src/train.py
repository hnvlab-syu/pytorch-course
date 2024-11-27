from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    loggers: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    if loggers:
        log.info("Logging hyperparameters...")
        log_hyperparameters(
            {
                "cfg": cfg,
                "datamodule": datamodule,
                "model": model,
                "callbacks": callbacks,
                "logger": loggers,
                "trainer": trainer,
            }
        )

    if cfg.get("train"):
        log.info("Starting training...")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics
    if cfg.get("test"):
        log.info("Starting testing...")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if not ckpt_path:
            log.warning("Best checkpoint not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    test_metrics = trainer.callback_metrics

    metrics = {**train_metrics, **test_metrics}

    return metrics, {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "loggers": loggers,
        "trainer": trainer,
    }


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    extras(cfg)

    metrics, _ = train(cfg)

    optimized_metric = cfg.get("optimized_metric")
    if optimized_metric:
        return get_metric_value(metric_dict=metrics, metric_name=optimized_metric)


if __name__ == "__main__":
    main()
