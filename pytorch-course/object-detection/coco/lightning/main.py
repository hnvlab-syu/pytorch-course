import os
import json
import argparse

import cv2
import numpy as np
from PIL import Image

import torch.optim as optim
from torchmetrics import MaxMetric
from torchmetrics.detection import MeanAveragePrecision

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar, DeviceStatsMonitor

# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval

from src.dataset import COCODataModule
from src.model import create_model
from src.utils import visualize


SEED = 36
L.seed_everything(SEED)

class DetectionModel(L.LightningModule):
    def __init__(
            self,
            model,
    ):
        super().__init__()
        self.model = model
        # self.batch_size = args.batch_size
        self.train_mAP = MeanAveragePrecision()
        self.val_mAP = MeanAveragePrecision()
        self.test_mAP = MeanAveragePrecision()

        # self.val_mAP_best = MaxMetric()
        # self.test_mAP_best = MaxMetric()

    def forward(self, images, targets):
        # if targets is None:  # predict 
        #     return self.model(images)
        return self.model(images, targets)

    def on_train_start(self):
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_mAP.reset()
        # self.val_mAP_best.reset()

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)   # multi-task loss(여러 개의 loss): 모델 내부에서 loss function 계산
        # print("Train--- Model loss_dict type:", type(loss_dict))
        # print("Train--- Model loss_dict:", loss_dict)
        loss = sum(loss for loss in loss_dict.values())

        # predictions = torch.argmax(loss_dict, dim=1)
        # visualize(images, targets, predictions)

        # outputs = self.model(images)
        # print("Train--- Model output type:", type(outputs))
        # print("Train--- Model output:", outputs)
        # predictions = {
        #     'boxes': outputs['boxes'],
        #     'scores': outputs['scores'],
        #     'labels': outputs['labels']
        #     # 모델이 반환하는 prediction 정보에 따라 조정
        # }
        # self.train_mAP.update(outputs, targets)

        self.log('train/total_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/loss_classifier', loss_dict['loss_classifier'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/loss_box_reg', loss_dict['loss_box_reg'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/loss_objectness', loss_dict['loss_objectness'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/loss_rpn_box_reg', loss_dict['loss_rpn_box_reg'], on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        # metric_compute = self.train_mAP.compute()
        # map = metric_compute['map'].numpy().tolist()
        # map_50 = metric_compute['map_50'].numpy().tolist()
        # map_75 = metric_compute['map_75'].numpy().tolist()
        
        # self.log('train/mAP', map, on_step=False, on_epoch=True, prog_bar=True)
        # self.log('train/mAP_50', map_50, on_step=False, on_epoch=True, prog_bar=True)
        # self.log('train/mAP_75', map_75, on_step=False, on_epoch=True, prog_bar=True)
        pass

    def validation_step(self, batch, batch_idx):
        # print(f'batch #{batch_idx}', batch)
        # print(f"\nValidation Step #{batch_idx}")
        images, targets = batch
        # loss_dict = self.model(images, targets)
        # print("Val--- Model loss_dict type:", type(loss_dict))
        # print("Val--- Model loss_dict:", loss_dict)
        # loss = sum(loss for loss in loss_dict.values())

        outputs = self.model(images)
        # print("Val--- Model output type:", type(outputs))
        # print("Val--- Model output:", outputs)
        # predictions = {
        #     'boxes': loss_dict['boxes'],
        #     'scores': loss_dict['scores'],
        #     'labels': loss_dict['labels']
        #     # 모델이 반환하는 prediction 정보에 따라 조정
        # }
        # predictions = torch.argmax(outputs, dim=1)
        visualize(images, targets, outputs, args.annots, args.save_path, batch_idx)
        self.val_mAP.update(outputs, targets)

        # self.log('val/total_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        # self.log('val/loss_classifier', loss_dict['loss_classifier'], on_step=True, on_epoch=True, prog_bar=True)
        # self.log('val/loss_box_reg', loss_dict['loss_box_reg'], on_step=True, on_epoch=True, prog_bar=True)
        # self.log('val/loss_objectness', loss_dict['loss_objectness'], on_step=True, on_epoch=True, prog_bar=True)
        # self.log('val/loss_rpn_box_reg', loss_dict['loss_rpn_box_reg'], on_step=True, on_epoch=False, prog_bar=True)
    
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
        # forward pass 한 번만 실행
        # loss_dict = self.model(images, targets)
        # loss = sum(loss for loss in loss_dict.values())
        outputs = self.model(images)
        # predictions = {
        #     'boxes': loss_dict['boxes'],
        #     'scores': loss_dict['scores'],
        #     'labels': loss_dict['labels']
        # }
        self.test_mAP.update(outputs, targets)

        # self.log('test/total_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        # self.log('test/loss_classifier', loss_dict['loss_classifier'], on_step=True, on_epoch=True, prog_bar=True)
        # self.log('test/loss_box_reg', loss_dict['loss_box_reg'], on_step=True, on_epoch=True, prog_bar=True)
        # self.log('test/loss_objectness', loss_dict['loss_objectness'], on_step=True, on_epoch=True, prog_bar=True)
        # self.log('test/loss_rpn_box_reg', loss_dict['loss_rpn_box_reg'], on_step=True, on_epoch=False, prog_bar=True)

    def on_test_epoch_end(self, outputs):
        metric_compute = self.test_mAP.compute()
        map = metric_compute['map'].numpy().tolist()
        map_50 = metric_compute['map_50'].numpy().tolist()
        map_75 = metric_compute['map_75'].numpy().tolist()
        
        self.log('test/mAP', map, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/mAP_50', map_50, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/mAP_75', map_75, on_step=False, on_epoch=True, prog_bar=True)

    # def predict_step(self, batch):
    #     images, _ = batch
    #     # outputs = self.model(images)
    #     # return outputs
    #     return self.model(images)
    
    def predict_step(self, batch, batch_idx):
        # images, _ = batch
        # if not isinstance(images, (list, tuple)):
        #     images = [images]  # 단일 이미지를 리스트로 변환
        # predictions = self.model(images)
        # return predictions  # boxes, labels, scores 반환
        images, _ = batch
        return self.model(images)  # 이미지를 리스트로 변환하는 부분 제거

    def configure_optimizers(self):
        return optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0005)
        # return optim.Adam(self.model.parameters(), lr=1e-4)


def main(batch_size, device, gpus, epoch, precision, num_workers, data, annots, detection_model, mode, ckpt, save_path):
    print('----------------------')
    print("Starting main function")
    print(f"Mode: {mode}")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"----------Created save directory: {save_path}")

    if device == 'gpu':
        if len(gpus) == 1:
            gpus = [int(gpus)]
        else:
            gpus = list(map(int, gpus.split(',')))
    elif device == 'cpu':
        gpus = 'auto'
        precision = 32

    if mode in ['predict']:
        batch_size = 1

    datamodule = COCODataModule(
        data_path = data, 
        annots_path = annots, 
        batch_size = batch_size,
        num_workers = num_workers,
        mode = mode
    )
    print("----------DataModule initialized")

    if mode == 'train':
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        callbacks = [
            ModelCheckpoint(
                monitor='val/mAP',
                dirpath=save_path,
                filename='best-{epoch}-{val_mAP:.2f}',
                save_top_k=1,           # best 모델만 
                mode='max',
                save_weights_only=True,
                # callback_state_key='best_checkpoint'
            ),
            # ModelCheckpoint(
            #     monitor='val/mAP',
            #     dirpath=save_path,
            #     filename='{epoch}-{val_mAP:.2f}',
            #     save_top_k=-1,
            #     mode='max',
            #     save_weights_only=True,
            #     callback_state_key='all_checkpoints'
            # ),
            EarlyStopping(
                monitor='val_mAP',
                min_delta=0.00,
                patience=10,
                verbose=False,
                mode='max'
            ),
            RichProgressBar(),
            # DeviceStatsMonitor()
        ]
        
        trainer = L.Trainer(
            accelerator = device,
            devices = gpus,
            max_epochs = epoch,
            precision = precision,
            logger = WandbLogger(project="object-detection",),
            callbacks = callbacks
            # callbacks=[checkpoint_callback]
            # strategy = 'ddp_find_unused_parameters_false',
        )

        model = DetectionModel(create_model(detection_model))
        print("----------Model created")
        trainer.fit(model, datamodule)
        trainer.test(model, datamodule, ckpt_path="path/to/best-*.ckpt")

    # elif mode == 'test':   # predict
    #     trainer = pl.Trainer(
    #         accelerator = device,
    #         devices = gpus,
    #         precision = precision
    #     )
    #     model = DetectionModel(create_model(detection_model)).load_from_checkpoint(ckpt)
    #     test_output = trainer.test(model, datamodule)
    #     outputs = []
    #     for i in range(len(test_output)):
    #         outputs += test_output[i]
    #     # outputs = outputs[0]+ outputs[1]
    #     print(len(outputs))
        
    #     # with open(args.valid_annt_path, "r") as f:
    #     #     annots = json.load(f)
    #     with open(os.path.join(annots, "instances_val2017.json"), 'r') as f:
    #         annots = json.load(f)

    #     images = annots['images']

    #     image_ids = []
    #     for image in images:
    #         image_id = image['id']
    #         image_ids.append(image_id)

    #     # eval
    #     predicts = []
    #     idx = 0
    #     for i in range(len(outputs)):
    #         # print(outputs[i])
    #         boxes = outputs[i]['boxes'].detach().cpu().numpy().tolist()
    #         scores = outputs[i]['scores'].detach().cpu().numpy().tolist()
    #         labels = outputs[i]['labels'].detach().cpu().numpy().tolist()

    #         for bbox,label,score in zip(boxes,labels,scores):
    #             # print(bbox,label,score)
    #             bbox[2] = bbox[2] - bbox[0]
    #             bbox[3] = bbox[3] - bbox[1]
    #             tmp = {"image_id": int(image_ids[idx]), "category_id": int(label), "bbox": bbox, "score": float(score)}
    #             predicts.append(tmp)
    #         idx += 1

    #     with open('predict.json', 'w') as f:
    #         json.dump(predicts, f)
        
    #     # coco_gt = COCO(args.valid_annt_path)
    #     coco_gt = COCO(os.path.join(annots, "instances_val2017.json"))
    #     coco_pred = coco_gt.loadRes('predict.json')
    #     coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')

    #     coco_eval.evaluate()
    #     coco_eval.accumulate()
    #     coco_eval.summarize()


    # else:   # predict
    #     trainer = L.Trainer(
    #         accelerator = device,
    #         devices = gpus,
    #         precision = precision
    #     )
    #     model = DetectionModel.load_from_checkpoint(checkpoint_path=ckpt, model=create_model(detection_model))
    #     predict_output = trainer.predict(model, datamodule)
        
    #     save_dir = os.path.join(save, 'predict_output')
    #     os.makedirs(save_dir, exist_ok=True)
        
    #     for i, output in enumerate(predict_output):
    #         # print('---------------------shape---------------------')
    #         # print(output.shape)     
    #         img_np = output.cpu().numpy().squeeze().transpose(1, 2, 0)   
    #         # print(img_np)
    #         img_np = (img_np * 255).clip(0, 255).astype(np.uint8)  
    #         # print(img_np)
    
    #         im = Image.fromarray(img_np)  
    #         im.save(os.path.join(save_dir, f'output.png'))

    elif mode == 'predict':
        trainer = L.Trainer(
            accelerator = device,
            devices = gpus,
            precision = precision
        )
        model = DetectionModel.load_from_checkpoint(
            checkpoint_path=ckpt,
            model=create_model(detection_model)
        )
        predictions = trainer.predict(model, datamodule)
        
        save_dir = os.path.join(save_path, 'predict_output')
        os.makedirs(save_dir, exist_ok=True)

        with open(os.path.join(annots, "instances_val2017.json"), 'r') as f:
            val_annots = json.load(f)
        categories = val_annots['categories']
        
        for i, pred in enumerate(predictions):
            if isinstance(pred, list):  # batch의 결과인 경우
                pred = pred[0]  # 첫 번째 이미지

            # 원본 이미지 로드
            img = Image.open(datamodule.pred_dataset[i])
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
                    
            output_img = Image.fromarray(img_np)
            output_img.save(os.path.join(save_dir, f'pred_{i}.png'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Faster-RCNN')
    parser.add_argument('-b', '--batch_size', type=int, default=4)
    parser.add_argument('-dc', '--device', type=str, default='gpu')
    parser.add_argument('-g', '--gpus', type=str, nargs='+', default='0')   # type=list, default=[0,1,2,3]
    parser.add_argument('-e', '--epoch', type=int, default=150)   # max_epochs
    parser.add_argument('-p', '--precision', type=str, default='16-mixed')   # 32-true/ 16-mixed
    parser.add_argument('-n', '--num_workers', type=int, default=0)
    parser.add_argument('-d', '--data_path', dest='data', type=str, default='../datasets/val2017')   # coco경로에서 python lightning/main.py
    parser.add_argument('-a', '--annots_path', dest='annots', type=str, default='../datasets/annotations')
    parser.add_argument('-m', '--model', type=str, default='fasterrcnn')
    parser.add_argument('-mo', '--mode', type=str, default='train')
    parser.add_argument('-c', '--ckpt', type=str, default='')
    parser.add_argument('-s', '--save_path', type=str, default='./checkpoint/')
    args = parser.parse_args()
    
    # if args.mode == 'train':
    #     wandb_logger = WandbLogger(project=args.project)
    #     metric = MeanAveragePrecision()
        
    main(args.batch_size, args.device, args.gpus, args.epoch, args.precision, args.num_workers, args.data, args.annots, args.model, args.mode, args.ckpt, args.save_path)
    # main(args)