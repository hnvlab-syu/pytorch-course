# 데이터셋
COCO 데이터셋의 Validation 데이터셋을 사용함

## 가상환경 생성 및 라이브러리 설치
- `environment.yaml` 파일의 `name` 부분을 원하는 가상환경 이름으로 변경할 것 (default: `instance`)
- 오류 발생 시 윈도우 환경에서 실행 할 것
- 라이브러리 관련 문제 발생 시 직접 라이브러리 설치할 것
```shell
conda env create -f environment.yaml
```

## 다운로드
- 접속 권한 확인 후 HnVLab Synology 접속 [Here](https://hnvlab.synology.me:5001/)
- COCO 폴더의 val2017.zip과 annotations_trainval2017.zip의 instances_val2017.json 파일을 다운
- 현재 경로의 dataset 폴더에 압축 해제 및 파일 배치
- 다음과 같은 형식이 되어야 함
```shell
|-- dataset
`-- |-- instances_val2017.json
    |-- instance_example.jpg
    `-- val2017
        |-- 000000000139.jpg
        |-- 000000000285.jpg
        |-- 000000000632.jpg
        |-- ... 
```

# 진행 방식
같은 분류 작업을 세 가지 방식으로 구현함
1. PyTorch
2. PyTorch Lightning
3. Lightning-Hydra-Template

자세한 내용은 각 디렉토리 내에 있는 README 파일에서 확인
