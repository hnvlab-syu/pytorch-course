from typing import List

import os
import cv2
import json
import cv2
import numpy as np
from PIL import Image

import hydra
import rootutils
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
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
    predictions = trainer.predict(model, datamodule)

    with open(os.path.join(cfg.paths.root_dir, "../datasets/annotations/instances_val2017.json"), 'r') as f:
        val_annots = json.load(f)
    categories = val_annots['categories']
     
    pred = predictions[0][0]

    img = Image.open(datamodule.pred_dataset[0])
    img_np = np.array(img)
    
    for box, category_id, score in zip(pred['boxes'], pred['labels'], pred['scores']):
        if score > 0.75:  # confidence threshold
            box = box.cpu().numpy()
            cv2.rectangle(
                img_np,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                (0, 0, 255),
                2   # 글씨 두께
            )

            category_name = None
            for n in categories:
                if n['id'] == category_id.item():  # tensor -> python int
                    category_name = n['name']
                    break
            if category_name is None:
                category_name = 'unknown'

            text = f"{category_name}({category_id}): {(score*100):.2f}"
            cv2.putText(img_np, text, (int(box[0]), int(box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX,   # OpenCV 이미지 좌표계: (0,0)왼쪽 상단 => y좌표 방향 반대
                            0.5, (0, 0, 255), 1, cv2.LINE_AA)
            
    save_dir = os.path.join(cfg.paths.output_dir, 'prediction')
    os.makedirs(save_dir, exist_ok=True)        

    output_img = Image.fromarray(img_np)
    output_img.save(os.path.join(save_dir, f'prediction.png'))

@hydra.main(version_base="1.3", config_path="../configs", config_name="predict.yaml")
def main(cfg: DictConfig) -> None:

    predict(cfg)


if __name__ == "__main__":
    main()
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