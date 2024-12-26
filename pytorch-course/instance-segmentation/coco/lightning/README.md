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
python lightning/main.py -m your_model_name -b 32 -e 5 \
  -d your_dataset_path \
  -a your_annotation_path \
  -s ./checkpoint/ -mo train
```

예측
```
python lightning/main.py -m your_model_name \
  -d your_dataset_path \
  -a your_annotation_path \
  -c your_checkpoint_path \
  -s ./checkpoint/ -mo predict
```
인자는 코드에서 확인해주세요

