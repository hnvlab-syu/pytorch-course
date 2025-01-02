# Hydra-template Semantic Segmentation

## 가상환경 생성 및 라이브러리 설치
- `semantic-segmentation/pascal/hydra-template`에서 사용되는 `environment.yaml`이므로 상위 디렉토리의 `environment.yaml` 파일과 구분하여 사용할 것
- `environment.yaml` 파일의 `name` 부분을 원하는 가상환경 이름으로 변경할 것(default: `semant`)
```shell
conda env create -f environment.yaml
```

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