# 데이터셋
COCO Dataset의 val2017을 사용

## 환경설정
1. environment.yaml을 통해 conda 가상환경 생성 및 패키지 설치 (권장) 
2. requirements.txt를 통해 pip 패키지 설치

## 다운로드
- [Here](https://cocodataset.org/#download)
- 2017 Val images, 2017 Train/Val annotations 파일을 다운로드
- 현재경로의 dataset 폴더에 압축해제
- 다음과 같은 형식이 되어야 함
```shell
|-- dataset
`-- |-- annotations
    |   |-- instances_val2017.json
    |-- val2017
    |   |-- 000000000139.jpg
    |   |-- 000000000285.jpg
    |   |-- 000000000632.jpg
    |   |-- ... 
    |-- ...
```

# 진행 방식
같은 분류 작업을 세 가지 방식으로 구현함
1. PyTorch
2. PyTorch Lightning
3. Lightning-Hydra-Template

자세한 내용은 각 디렉토리 내에 있는 README 파일에서 확인
