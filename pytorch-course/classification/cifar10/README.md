### 1. 데이터 다운로드 및 경로 지정 방법
1. https://www.kaggle.com/competitions/cifar-10/data 에서 train.7z와 trainLabels.csv 다운로드
2. train.7z를 pytorch-course/classification/cifar10/data 경로로 압축해제
3. trainLabels.csv를 다운받은 경로에서 pytorch-course/classification/cifar10/data (train.py와 test.py가 있는 폴더 기준 ./data)로 경로 이동

### 2. 실행 방법
- 주의: pytorch-course/classification/cifar10 (train.py와 test.py가 있는 폴더)에서 실행해야 함.
- GPU를 활용해 학습 (train)
    - Windows 10/11 혹은 Linux (Ubuntu)
        ```
        python train.py --device cuda
        ```
    - IOS/Mac OS
        ```
        python train.py --device nps
        ```
- GPU를 활용해 추론 (test)
    - Windows 10/11 혹은 Linux (Ubuntu)
        ```
        python test.py --device cuda
        ```
    - IOS/Mac OS
        ```
        python test.py --device nps
        ```