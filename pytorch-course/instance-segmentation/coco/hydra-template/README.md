# pytorch lightning instance segmentation

##COCO Dataset을 사용하여 Instance Segmentation 모델을 학습

## folder 구조
```
coco/
├── hydra-template/ 
│ ├── configs/ 
│ ├── notebooks/ 
│ ├── scripts/  
│ └── src/ 
├── lightning/
│ ├── src/ 
│ ├── main/ 
│ └── utils/
├── datasets/
└── requirements.txt 
```

## environment
- ubuntu 20.04 에서 진행하는것을 권장

1. 필수 라이브러리 설치 
```
pip install -r requirements.txt

```
(오류가 발생한다면 필요한 라이브러리 직접 설치해주세요)

2. 코드실행

학습
```
python ./hydra-template/src/train.py
python ./hydra-template/src/train.py src/train.py experiment=train.yaml
python ./hydra-template/src/train.py +trainer.precision='16-mixed' trainer.max_epochs=30 trainer=gpu model.net.model=mask_rcnn seed=36 data.batch_size=128 callbacks.early_stopping.patience=5

# 다양한 파라미터들로 여러번 학습할 때 실행
bash scripts/schedule.sh

```

예측
```
python ./hydra-template/src/predict.py experiment=predict.yaml

python ./hydra-template/src/predict.py trainer=gpu model.net.model=mask_rcnn ckpt_path="your_ckpt_path" data.mode="predict" data.data_dir="example.jpg"

```
ckpt_path: 학습된 모델의 ckpt 경로
data.mode: 추론할땐 반드시 predict으로 변경
data.data_dir: 추론하고 싶은 이미지 경로