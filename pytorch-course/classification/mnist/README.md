### 1. 실행 방법
- 주의: pytorch-course/classification/mnist (train.py와 test.py가 있는 폴더)에서 실행해야 함.
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