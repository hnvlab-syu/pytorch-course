# Pytorch Super-Resolution
## Download Datasets
Ref. <https://data.vision.ee.ethz.ch/cvl/DIV2K/> 
Download Path: /super-resolution/DIV2K/datasets/

- (NTIRE 2018) Low Res Images:
    - Train Data Track 2 realistic mild x4 (LR images)
    - Validation Data Track 2 realistic mild x4 (LR images)
- High Resolution Images:
    - Train Data (HR images)
    - Validation Data (HR images)


## BasicSR
Ref. <https://github.com/XPixelGroup/BasicSR.git>    ---docs/INSTALL.md
- git clone 후, submodule 적용
```
git submodule init
git submodule update
pip install basicsr
```

## DIV2K Preparation Steps
Ref. BasicSR/docs/DatasetPreparation.md

```
# 이미지 경로와 crop_size 확인 후
python extract_subimages.py 
```
- /super-resolution/DIV2K/datasets/ 경로에 각 데이터셋의_sub 디렉토리 생성 여부 확인


## Train
```
python main.py
```


## Test
```
python main.py -lrd ../datasets/DIV2K_valid_LR_mild -hrd ../datasets/DIV2K_valid_HR -mo test
```
- dataset 경로와 mode 설정


## Predict
```
python main.py -lrd example.jpg -mo predict 
```
- 이미지 경로와 mode 설정