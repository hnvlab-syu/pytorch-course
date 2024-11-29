import os
import argparse
import numpy as np
import cv2 as cv2
import lightning as L

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar, DeviceStatsMonitor
from lightning.pytorch.loggers import WandbLogger

from src.dataset import DIV2KDataModule
from src.model import create_model, SRModel


SEED = 36
L.seed_everything(SEED)


def main(model, weights, lr_data, hr_data, learning_rate, batch_size, num_workers, epoch, save, device, gpus, precision, mode, ckpt):
    sr_model = SRModel(create_model(model, weights), learning_rate, batch_size)
    
    if not os.path.exists(save):
        os.makedirs(save)

    if device == 'gpu':
        if len(gpus) == 1:
            gpus = [int(gpus)]
        else:
            gpus = list(map(int, gpus.split(',')))
    elif device == 'cpu':
        gpus = 'auto'
        precision = 32

    if mode in ['test', 'predict']:
        batch_size = 1
    datamodule = DIV2KDataModule(
        lr_dir = lr_data,
        hr_dir = hr_data,
        batch_size = batch_size,
        num_workers= num_workers,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=save,
            filename=f'sr_model-'+'{epoch:02d}-{val_psnr:.2f}',
            monitor='val_psnr',
            mode='max',
            save_top_k=3,
        ),
        EarlyStopping(
            monitor='val_psnr',
            mode='max',
            patience=10
        ),
        RichProgressBar(),
        DeviceStatsMonitor()
    ]

    trainer = L.Trainer(
        accelerator = device,
        devices = gpus if device == 'gpu' else None,
        max_epochs = epoch,
        precision = precision,
        logger = WandbLogger(project="SR",),
        callbacks = callbacks,
        gradient_clip_val = 0.5,
    )

    if mode == 'train':
        trainer.fit(sr_model, datamodule)
        
    elif mode == 'test':
        # trainer.test(sr_model, datamodule, ckpt_path=ckpt)
        sr_model = SRModel.load_from_checkpoint(ckpt, model=create_model(model, weights))
        test_results = trainer.test(sr_model, datamodule)

        # Save test results
        save_dir = os.path.join(save, 'test_results')
        os.makedirs(save_dir, exist_ok=True)
        
        # Save metrics to a file
        with open(os.path.join(save_dir, 'metrics.txt'), 'w') as f:
            for metric, value in test_results[0].items():
                f.write(f"{metric}: {value}\n")

    else:  # predict/inferece
        sr_model = SRModel.load_from_checkpoint(ckpt, model=create_model(model, weights))
        predictions = trainer.predict(sr_model, datamodule)
        
        # Save predictions
        save_dir = os.path.join(save, 'predictions')
        os.makedirs(save_dir, exist_ok=True)
        
        for i, pred in enumerate(predictions):
            img_np = pred[0].cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(save_dir, f'pred_{i}.png'),
                cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='edsr')
    parser.add_argument('-w', '--weights', type=str, default='./weights/EDSR/EDSR_Mx4_f64b16_DIV2K_official-0c287733.pth')
    parser.add_argument('-lrd', '--lr_data_path', dest='lr_data', type=str, default='../datasets/DIV2K_train_LR_mild_sub')
    parser.add_argument('-hrd', '--hr_data_path', dest='hr_data', type=str, default='../datasets/DIV2K_train_HR_sub')     # '/DIV2K/dataset/DIV2K_valid_HR'
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-b', '--batch_size', type=int, default=8)       # 1/ dest='batch'
    parser.add_argument('-n', '--num_workers', type=int, default=4)       # 0
    parser.add_argument('-e', '--epoch', type=int, default=150)
    parser.add_argument('-s', '--save_path', dest='save', type=str, default='./checkpoint/')
    parser.add_argument('-dc', '--device', type=str, default='gpu')
    parser.add_argument('-g', '--gpus', type=str, nargs='+', default='0')
    parser.add_argument('-p', '--precision', type=str, default='16-mixed')  # 32-true/ 16-mixed
    parser.add_argument('-mo', '--mode', type=str, default='train')
    parser.add_argument('-c', '--ckpt_path', dest='ckpt', type=str, default='./checkpoint/sr_model-epoch=21-val_psnr=18.09.ckpt')
    args = parser.parse_args()

    main(args.model, args.weights, args.lr_data, args.hr_data, args.learning_rate, args.batch_size, args.num_workers, 
         args.epoch, args.save, args.device, args.gpus, args.precision, args.mode, args.ckpt)  
    


    

