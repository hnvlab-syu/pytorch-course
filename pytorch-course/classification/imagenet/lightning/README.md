# Pytorch Classification
## 훈련
주의: 현재 디렉토리에서 아래 명령어 사용해야 함
```shell
python train.py -m your_device -dc "your_image_path -g "0"
```
- model(필수): 사용할 모델 선택 (vgg or resnet or efficientnet)
- device(선택): 훈련에 사용할 디바이스 선택 (cpu or cuda)
- 나머지 인자는 코드에서 확인
## 예측
주의: 현재 디렉토리에서 아래 명령어 사용해야 함
```shell
python predict.py -m your_device -dc "your_image_path -g "0" --image_path "your_image_path
```
- ip, image_path(필수): 예측할 이미지 파일 선택
- m, model(필수): 사용할 모델 선택 (vgg or resnet or efficientnet)
- dc, device(선택): 훈련에 사용할 디바이스 선택 (cpu or gpu)
- g, gpus(선택): 훈련에 사용할 gpu개수 선택
- 나머지 인자는 코드에서 확인