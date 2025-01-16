# PyTorch Lightning: Object-Detection
## 훈련
주의: 현재 디렉토리에서 아래 명령어 사용해야 함
```shell
python main.py
```
- -d, --data_path(선택): 학습할 이미지 파일 선택 (Default: ../datasets/val2017)
- -a, --annots_path(선택): 학습할 이미지의 annotations 파일 선택 (Default: ../datasets/annotations)
- -mo, --mode(선택): 모델의 모드 선택 (Default: train; train or predict)
- -m, --model(선택): 사용할 모델 선택 (Default: fasterrcnnv2; fasterrcnn or fasterrcnnv2)
- -dc, --device(선택): 훈련에 사용할 디바이스 선택 (cpu or cuda)
- 나머지 인자는 코드에서 확인


## 예측
주의: 현재 디렉토리에서 아래 명령어 사용해야 함
```shell
python main.py -d ../dataset/instance_example.jpg -mo predict
python predict.py --data_path "your_image_path" --model "your_model" --mode predict
```
- -d, --data_path(필수): 예측할 이미지 파일 선택
- -mo, --mode(필수): 모델의 모드 선택 (predict)
- -m, --model(선택): 사용할 모델 선택 (Default: fasterrcnnv2; fasterrcnn or fasterrcnnv2)
- -dc, --device(선택): 훈련에 사용할 디바이스 선택 (cpu or gpu)
- -g, --gpus(선택): 훈련에 사용할 gpu개수 선택
- 나머지 인자는 코드에서 확인