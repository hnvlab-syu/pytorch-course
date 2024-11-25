### 1. 데이터 다운로드 및 경로 지정 방법
1. https://drive.google.com/file/d/14VpWSy1yYZG8p39FrrgiWTj9HtKWMzYK/view?usp=sharing 을 클릭하여 다운로드
2. 를 pytorch-course/classification/fashion-mnist/data (train.py와 test.py가 있는 폴더 기준 ./data) 경로로 압축해제

### 2. 실행 방법
- 주의: pytorch-course/classification/fashion-mnist (train.py와 test.py가 있는 폴더)에서 실행해야 함.
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