# Pytorch Classification
## 라이브러리
Lightning-Hydra-Template을 기반으로 작성 [Here](https://github.com/ashleve/lightning-hydra-template)
## configs 설정
```shell
├── configs                   <- Hydra configs
│   ├── callbacks                <- Callbacks configs
│   ├── data                     <- Data configs
│   ├── experiment               <- Experiment configs
│   ├── extras                   <- Extra utilities configs
│   ├── hydra                    <- Hydra configs
│   ├── logger                   <- Logger configs
│   ├── model                    <- Model configs
│   ├── paths                    <- Project paths configs
│   ├── trainer                  <- Trainer configs
│   │
│   ├── predict.yaml             <- Main config for prediction
│   └── train.yaml               <- Main config for training
```
## 훈련
주의: 현재 디렉토리에서 아래 명령어 사용해야 함
```shell
python src/train.py

python src/train.py experiment=train.yaml

python src/train.py +trainer.precision='16-mixed' trainer.max_epochs=30 trainer=gpu model.net.model=efficientnet seed=36 data.batch_size=128 callbacks.early_stopping.patience=5

# 다양한 파라미터들로 여러번 학습할 때 실행
bash scripts/schedule.sh
```
- trainer: gpu, cpu, ddp 및 학습 paramerts 설정
## 예측
주의: 현재 디렉토리에서 아래 명령어 사용해야 함
```shell
python src/predict.py experiment=predict.yaml

python src/predict.py trainer=gpu model.net.model=efficientnet ckpt_path="your_ckpt_path" data.mode="predict" data.data_dir="example.jpg"
```
- ckpt_path: 학습된 모델의 ckpt 경로
- data.mode: 추론할땐 반드시 predict으로 변경
- data.data_dir: 추론하고 싶은 이미지 경로