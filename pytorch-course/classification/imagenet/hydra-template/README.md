# Pytorch Classification
## 라이브러리
Lightning-Hydra-Template을 기반으로 작성 [Here]https://github.com/ashleve/lightning-hydra-template

## configs 설정
```shell
├── configs                   <- Hydra configs
│   ├── callbacks                <- Callbacks configs
│   ├── data                     <- Data configs
│   ├── debug                    <- Debugging configs
│   ├── experiment               <- Experiment configs
│   ├── extras                   <- Extra utilities configs
│   ├── hparams_search           <- Hyperparameter search configs
│   ├── hydra                    <- Hydra configs
│   ├── local                    <- Local configs
│   ├── logger                   <- Logger configs
│   ├── model                    <- Model configs
│   ├── paths                    <- Project paths configs
│   ├── trainer                  <- Trainer configs
│   │
│   ├── eval.yaml             <- Main config for evaluation
│   └── train.yaml            <- Main config for training
```

## 훈련
주의: 현재 디렉토리에서 아래 명령어 사용해야 함
```shell
python src/train.py

python src/train.py experiment=train.yaml

python src/train.py +trainer.precision='16-mixed' trainer.max_epochs=30 trainer=gpu model.net.model=efficientnet seed=36 data.batch_size=128 callbacks.early_stopping.patience=5

bash scripts/schedule.sh
```
- trainer: gpu, cpu, ddp 및 학습 paramerts 설정
- 

## 예측
주의: 현재 디렉토리에서 아래 명령어 사용해야 함
```shell
python src/predict.py experiment=predict.yaml

python src/predict.py trainer=gpu model.net.model=efficientnet ckpt_path="logs\train\runs\2024-11-08_02-35-08\checkpoints\epoch_005.ckpt" data.mode="predict" data.data_dir="cat.jpg"
```