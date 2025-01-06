# 데이터셋
ImageNet 데이터셋의 Validation 데이터셋을 사용함

## 환경설정
1. environment.yaml을 통해 conda 가상환경 생성 및 패키지 설치
2. requirements.txt를 통해 pip 패키지 설치

## 다운로드
- 접속 권한 확인 후 HnVLab Synology 접속 [Here](https://hnvlab.synology.me:5001/)
- Imagenet-val.zip 파일을 다운로드
- 현재경로의 dataset 폴더에 압축해제
- 다음과 같은 형식이 되어야 함
```shell
|-- dataset
`-- |-- folder_num_class_map.txt
    |-- n01440764
    |   |-- ILSVRC2012_val_00000293.JPEG
    |   |-- ILSVRC2012_val_00002138.JPEG
    |   |-- ILSVRC2012_val_00003014.JPEG
    |   |-- ... 
    |-- n01443537
    |   |-- 000000000236.jpg
    |   |-- 000000000262.jpg
    |   |-- 000000000307.jpg
    |   |-- ... 
    |-- ...
```

# 진행 방식
같은 분류 작업을 세 가지 방식으로 구현함
1. PyTorch
2. PyTorch Lightning
3. Lightning-Hydra-Template

자세한 내용은 각 디렉토리 내에 있는 README 파일에서 확인
