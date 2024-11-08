### 1. 데이터 다운로드 및 경로 지정 방법
1. http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit 에서 training/validation data 다운로드
2. pytorch-course/semantic-segmentation/pascal-voc-2012/data 경로에 압축파일 내의 VOC2012 파일 압축해제
   디렉토리 결과: semantic-segmentation/pascal-voc-2012/data/VOC2012/*

### 2. 실행 방법
- 주의: pytorch-course/semantic-segmentation/pascal-voc-2012(train.py와 test.py가 있는 폴더)에서 실행해야 함.
- 학습 및 추론 기본 설정은 cuda (없을 경우 cpu)