from typing import Any, Dict

import warnings
import torch
import lightning.pytorch as L
# from torchmetrics import MaxMetric
from torchmetrics.detection import MeanAveragePrecision
# from src.utils import visualize

warnings.filterwarnings('ignore')


SEED = 36
L.seed_everything(SEED)

class DetectionModel(L.LightningModule):
    def __init__(
            self,
            net: torch.nn.Module,
            optimizer: torch.optim.Optimizer,    # torch.optim.Adam
            scheduler: torch.optim.lr_scheduler,
            compile: bool
    ):
        super().__init__()
        print(f"Initializing DetectionModel with net: {net}")
        print(type(net))
        self.save_hyperparameters(logger=False)
        self.net = net
        self.train_mAP = MeanAveragePrecision()
        self.val_mAP = MeanAveragePrecision()
        self.test_mAP = MeanAveragePrecision()

    def forward(self, images, targets):
        return self.net(images, targets)

    def on_train_start(self):
        self.val_mAP.reset()

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.net(images, targets)   # multi-task loss(여러 개의 loss): 모델 내부에서 loss function 계산

        loss = sum(loss for loss in loss_dict.values())

        self.log('train/total_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/loss_classifier', loss_dict['loss_classifier'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/loss_box_reg', loss_dict['loss_box_reg'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/loss_objectness', loss_dict['loss_objectness'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/loss_rpn_box_reg', loss_dict['loss_rpn_box_reg'], on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.net(images)

        # visualize(images, targets, outputs, args.annots, args.save_path, batch_idx)
        self.val_mAP.update(outputs, targets)
    
    def on_validation_epoch_end(self):
        metric_compute = self.val_mAP.compute()
        map = metric_compute['map'].numpy().tolist()
        map_50 = metric_compute['map_50'].numpy().tolist()
        map_75 = metric_compute['map_75'].numpy().tolist()
        
        self.log('val/mAP', map, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/mAP_50', map_50, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/mAP_75', map_75, on_step=False, on_epoch=True, prog_bar=True)
        self.val_mAP.reset()
    
    def test_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.net(images)

        self.test_mAP.update(outputs, targets)

    def on_test_epoch_end(self, outputs):
        metric_compute = self.test_mAP.compute()
        map = metric_compute['map'].numpy().tolist()
        map_50 = metric_compute['map_50'].numpy().tolist()
        map_75 = metric_compute['map_75'].numpy().tolist()
        
        self.log('test/mAP', map, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/mAP_50', map_50, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/mAP_75', map_75, on_step=False, on_epoch=True, prog_bar=True)
    
    def predict_step(self, batch, batch_idx):
        images = [batch]
        return self.net(images)  
    
    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.net.parameters())  # self.trainer.model.parameters()
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/mAP",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

if __name__ == "__main__":
    _ = DetectionModel(None, None, None, None)

# def main(batch_size, device, gpus, epoch, precision, num_workers, data, annots, detection_model, mode, ckpt, save_path):
#     print('----------------------')
#     print("Starting main function")
#     print(f"Mode: {mode}")

#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#         print(f"----------Created save directory: {save_path}")

#     if device == 'gpu':
#         if len(gpus) == 1:
#             gpus = [int(gpus)]
#         else:
#             gpus = list(map(int, gpus.split(',')))
#     elif device == 'cpu':
#         gpus = 'auto'
#         precision = 32

#     # if mode in ['predict']:
#     #     batch_size = 1

#     datamodule = COCODataModule(
#         data_dir = data, 
#         annots_dir = annots, 
#         batch_size = batch_size,
#         num_workers = num_workers,
#         mode = mode
#     )
#     print("----------DataModule initialized")

#     if mode == 'train':
#         if not os.path.exists(save_path):
#             os.makedirs(save_path)

#         callbacks = [
#             ModelCheckpoint(
#                 monitor='val/mAP',
#                 dirpath=save_path,
#                 filename='best-{epoch}-{val_mAP:.2f}',
#                 save_top_k=1,           # best 모델만 
#                 mode='max',
#                 save_weights_only=True,
#                 # callback_state_key='best_checkpoint'
#             ),
#             # ModelCheckpoint(
#             #     monitor='val/mAP',
#             #     dirpath=save_path,
#             #     filename='{epoch}-{val_mAP:.2f}',
#             #     save_top_k=-1,
#             #     mode='max',
#             #     save_weights_only=True,
#             #     callback_state_key='all_checkpoints'
#             # ),
#             EarlyStopping(
#                 monitor='val/mAP',
#                 min_delta=0.00,
#                 patience=10,
#                 verbose=False,
#                 mode='max'
#             ),
#             RichProgressBar(),
#             # DeviceStatsMonitor()
#         ]
        
#         trainer = L.Trainer(
#             log_every_n_steps = 1,
#             accelerator = device,
#             devices = gpus,
#             max_epochs = epoch,
#             precision = precision,
#             logger = WandbLogger(project="object-detection",),
#             callbacks = callbacks
#             # callbacks=[checkpoint_callback]
#             # strategy = 'ddp_find_unused_parameters_false',
#         )

#         model = DetectionModel(create_model(detection_model))
#         print("----------Model created")
#         trainer.fit(model, datamodule)
#         trainer.test(model, datamodule, ckpt_path="path/to/best-*.ckpt")

#     else:   # predict
#         trainer = L.Trainer(
#             accelerator = device,
#             devices = gpus,
#             precision = precision
#         )
#         model = DetectionModel.load_from_checkpoint(
#             checkpoint_path=ckpt,
#             model=create_model(detection_model)
#         )
#         predictions = trainer.predict(model, datamodule)
        
#         save_dir = os.path.join(save_path, 'predict_output')
#         os.makedirs(save_dir, exist_ok=True)

#         with open(os.path.join(annots, "instances_val2017.json"), 'r') as f:
#             val_annots = json.load(f)
#         categories = val_annots['categories']
        
#         for i, pred in enumerate(predictions):
#             if isinstance(pred, list):  
#                 pred = pred[0]  

#             img = Image.open(datamodule.pred_dataset[i])
#             img_np = np.array(img)
            
#             for box, category_id, score in zip(pred['boxes'], pred['labels'], pred['scores']):
#                 if score > 0.75:  # confidence threshold
#                     box = box.cpu().numpy()
#                     cv2.rectangle(
#                         img_np,
#                         (int(box[0]), int(box[1])),
#                         (int(box[2]), int(box[3])),
#                         (0, 0, 255),
#                         2   # 글씨 두께
#                     )

#                     category_name = None
#                     for n in categories:
#                         if n['id'] == category_id.item():  # tensor -> python int
#                             category_name = n['name']
#                             break
#                     if category_name is None:
#                         category_name = 'unknown'

#                     text = f"{category_name}({category_id}): {(score*100):.2f}"
#                     cv2.putText(img_np, text, (int(box[0]), int(box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX,   # OpenCV 이미지 좌표계: (0,0)왼쪽 상단 => y좌표 방향 반대
#                                     0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    
#             output_img = Image.fromarray(img_np)
#             output_img.save(os.path.join(save_dir, f'pred_{i}.png'))


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Faster-RCNN')
#     parser.add_argument('-b', '--batch_size', type=int, default=4)
#     parser.add_argument('-dc', '--device', type=str, default='gpu')
#     parser.add_argument('-g', '--gpus', type=str, nargs='+', default='0')   # type=list, default=[0,1,2,3]
#     parser.add_argument('-e', '--epoch', type=int, default=150)   # max_epochs
#     parser.add_argument('-p', '--precision', type=str, default='16-mixed')   # 32-true/ 16-mixed
#     parser.add_argument('-n', '--num_workers', type=int, default=0)
#     parser.add_argument('-d', '--data_path', dest='data', type=str, default='../datasets/val2017')   # coco경로에서 python lightning/main.py
#     parser.add_argument('-a', '--annots_path', dest='annots', type=str, default='../datasets/annotations')
#     parser.add_argument('-m', '--model', type=str, default='fasterrcnn')
#     parser.add_argument('-mo', '--mode', type=str, default='train')
#     parser.add_argument('-c', '--ckpt', type=str, default='')
#     parser.add_argument('-s', '--save_path', type=str, default='./checkpoint/')
#     args = parser.parse_args()
    
#     main(args.batch_size, args.device, args.gpus, args.epoch, args.precision, args.num_workers, args.data, args.annots, args.model, args.mode, args.ckpt, args.save_path)