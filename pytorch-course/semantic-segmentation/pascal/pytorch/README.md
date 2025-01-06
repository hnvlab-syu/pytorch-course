# PyTorch Semantic Segmentation

## 가상환경 생성 및 라이브러리 설치
상위 디렉토리의 `environment.yaml` 파일을 사용할 것

## 훈련 및 추론
주의: 현재 디렉토리에서 다음 명령어 사용할 것
```shell
python train.py
```
- 인자들을 확인할 것

## 테스트
주의: 현재 디렉토리에서 다음 명령어 사용할 것
```shell
python test.py -c "your_model_checkpoint"
```
- 인자들을 확인할 것

## 예측
주의: 현재 디렉토리에서 다음 명령어 사용할 것

predict는 이미지 하나를 예측하는 작업임
```shell
python predict.py -mo predict -d "your_image_path" -c "your_model_checkpoint"
```
- `-mo predict`: 수행할 작업 선택(`train` or `predict`) 
- `-d`: 예측에 사용할 이미지 경로(default: `'../dataset/example.jpg'`)
- `-c`: 예측에 사용할 모델 파라미터 체크포인트 경로
