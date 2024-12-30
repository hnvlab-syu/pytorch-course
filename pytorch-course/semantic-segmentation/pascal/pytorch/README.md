# Lightning Semantic Segmentation

## 가상환경 생성 및 라이브러리 설치
- `semantic-segmentation/pascal/pytorch` 및 `semantic-segmentation/pascal/lightning`에서 사용되는 `environment.yaml`이므로 상위 디렉토리의 `environment.yaml` 파일과 구분하여 사용할 것
- `environment.yaml` 파일의 `name` 부분을 원하는 가상환경 이름으로 변경할 것(default: `semant`)
```shell
conda env create -f environment.yaml
```

## 훈련 및 추론
주의: 현재 디렉토리에서 다음 명령어 사용할 것
```shell
python train.py
```
- 인자들을 확인할 것

## 테스트
주의: 현재 디렉토리에서 다음 명령어 사용할 것
```shell
python test.py -c your_model_checkpoint
```
- 인자들을 확인할 것

## 예측
주의: 현재 디렉토리에서 다음 명령어 사용할 것

predict는 이미지 하나를 예측하는 작업임
```shell
python main.py -mo predict -d your_image_path -c your_model_checkpoint
```
- `-mo predict`: 수행할 작업 선택(`train` or `predict`) 
- `-d`: 예측에 사용할 이미지 경로
- `-c`: 예측에 사용할 모델 파라미터 체크포인트 경로