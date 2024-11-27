# COCO 데이터셋의 validation 데이터셋을 사용
COCO Dataset을 사용하여 Instance Segmentation 모델을 학습

## folder 구조
## 다운로드
- 접속 권한 확인 후 HnVLab Synology 접속 [Here](https://hnvlab.synology.me:5001/)
- Imagenet-val.zip 파일을 자신의 공유 폴더에 복사
- 현재경로의 dataset 폴더에 압축해제
- 다음과 같은 형식이 되어야 함
```shell
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

같은 분류 작업을 세 가지 테스크로 구분함
=======
# PyTorch Course
삼육대학교 HnV 연구실에서 작성한 코드로, 학생들 교육을 위해 만들어졌습니다.

Pytorch Course에서 사용하는 라이브러리
>>>>>>> upstream/develop
1. PyTorch
2. PyTorch Lightning
3. Lightning-Hydra-Template

<<<<<<< HEAD
자세한 내용은 각 디렉토리 내에 있는 README 파일에서 확인

## Download Datasets
- 접속 권한 확인 후 HnVLab Synology 접속 [Here](https://hnvlab.synology.me:5001/)
- Synology/dataset/표준코드에서 각 테스크에 맞는 데이터셋 다운로드
- 각 테스크에 맞는 dataset 폴더에 압축해제

1. ImageNet val
2. COCO 2017 val
3. Pascal VOC 2012 val
4. DIV2K

## Computer Vision Tasks
1. Image classification
2. Object detection
3. semantic segmentation
4. Instance segmentation
5. Super resolution
