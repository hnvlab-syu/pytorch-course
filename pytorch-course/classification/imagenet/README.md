# 데이터셋
ImageNet 데이터셋의 Validation 데이터셋을 사용함
### 다운로드
- 접속 권한 확인 후 HnVLab Synology 접속 [Here](https://hnvlab.synology.me:5001/)
- 로컬 컴퓨터와의 공유 폴더 생성
- Synology/dataset에 있는 Imagenet-val.zip 파일을 자신의 공유 폴더에 복사
- 이 파일이 있는 폴더의 dataset 폴더에 압축해제
- 다음과 같은 형식이 되어야 함
```shell
|-- dataset
`-- |-- 
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
# 테스크
같은 분류 작업을 세 가지 테스크로 구분함
1. Pytorch
2. Lightning
3. Lightning + Hydra

자세한 내용은 각 디렉토리 내에 있는 README 파일에서 확인