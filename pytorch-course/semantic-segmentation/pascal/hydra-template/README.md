# Hydra-template Semantic Segmentation

## 가상환경 생성 및 라이브러리 설치
상위 디렉토리의 `environment.yaml` 파일을 사용할 것

## 훈련 및 추론
주의: 현재 디렉토리에서 다음 명령어 사용할 것
```shell
python src/train.py

# train with gpu
python src/train.py trainer=gpu.yaml
```
- 인자들을 확인할 것

## 예측
주의: 현재 디렉토리에서 다음 명령어 사용할 것
```shell
python src/predict.py ckpt_path="your_model_checkpoint" pred_image="your_image_path"

# predict with gpu
python src/predict.py trainer=gpu.yaml ckpt_path="your_model_checkpoint" pred_image="..your_image_path"
```