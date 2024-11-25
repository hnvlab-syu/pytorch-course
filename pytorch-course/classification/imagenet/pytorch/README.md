# Pytorch Classification

## 훈련
주의: 현재 디렉토리에서 아래 명령어 사용해야 함
```shell
python train.py --device "your_device" --model "your_model"
```
- model(필수): 사용할 모델 선택 (vgg or resnet or efficientnet)
- device(선택): 훈련에 사용할 디바이스 선택 (cpu or cuda)
- 나머지 인자는 코드에서 확인
## 예측
주의: 현재 디렉토리에서 아래 명령어 사용해야 함
```shell
python predict.py --device "your_device" --model "your_model" --image_path "your_image_path"
```
- image_path(필수): 예측할 이미지 파일 선택
- model(필수): 사용할 모델 선택 (vgg or resnet or efficientnet)
- device(선택): 훈련에 사용할 디바이스 선택 (cpu or cuda)
- 나머지 인자는 코드에서 확인