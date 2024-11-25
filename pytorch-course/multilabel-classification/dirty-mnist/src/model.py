from torch import nn, Tensor
from torchvision.models import resnet50


class MultiLabelResNet(nn.Module):
    """Dirty-MNIST 데이터를 훈련할 모델을 정의합니다.
    모델은 torchvision에서 제공하는 ResNet-50을 지원합니다.
    Model Link : https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
    """
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.resnet = resnet50()
        self.classifier = nn.Linear(1000, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """피드 포워드(순전파)를 진행하는 함수입니다.

        :param x: 입력 이미지
        :type x: Tensor
        :return: 입력 이미지에 대한 예측값
        :rtype: Tensor
        """
        x = self.resnet(x)
        x = self.classifier(x)

        return x
