import warnings
from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Tuple
from omegaconf import DictConfig
from src.utils import pylogger, rich_utils
import torch
import os
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt
log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: Dict[str, Any], metric_name: Optional[str]) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value

class ObjectKeypointSimilarity:
    def __init__(self, image_dir: os.PathLike, ann_file: os.PathLike) -> None:
        self.coco_gt = COCO(ann_file)
        self.detections = []

    def update(self, preds, image_ids):
        for pred in preds:
            
            bbox = np.array(pred['bbox'])
            if len(bbox.shape) == 1:
                bbox = bbox.reshape(1, -1)
            
           
            keypoints = np.array(pred['keypoints']).reshape(-1).tolist()
            
            self.detections.append({
                'image_id': pred['image_id'],
                'category_id': pred['category_id'],
                'bbox': bbox.flatten().tolist(),
                'keypoints': keypoints,
                'score': pred['score']
            })

    def reset(self):
        self.detections = []

    def compute(self):
        if len(self.detections) == 0:
            print("No detections to evaluate")
            return
            
        coco_dt = self.coco_gt.loadRes(self.detections)
        
        
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'keypoints')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()


def visualize_pose_estimation(image, output, save_path):
    keypoints = output.cpu().numpy()
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy().transpose(1, 2, 0)
        image = ((image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255).astype(np.uint8)
    
    
    edges = [
        (0, 1), (0, 2), (1, 3), (2, 4), 
        (5, 6), (5, 7), (6, 8),         
        (7, 9), (8, 10),                  
        (5, 11), (6, 12),                 
        (11, 13), (12, 14),              
        (13, 15), (14, 16)             
    ]
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0.5:  
            plt.scatter(x, y, c='r', s=50)
    
    for (i, j) in edges:
        if keypoints[i][2] > 0.5 and keypoints[j][2] > 0.5:
            plt.plot([keypoints[i][0], keypoints[j][0]], 
                    [keypoints[i][1], keypoints[j][1]], 'g-')
    
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()