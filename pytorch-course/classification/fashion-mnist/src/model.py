from torch import nn, Tensor


class NeuralNetwork(nn.Module):
    """FashionMNIST 데이터를 훈련할 모델을 정의합니다.
    """
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """피드 포워드(순전파)를 진행하는 함수입니다.

        :param x: 입력 이미지
        :type x: Tensor
        :return: 입력 이미지에 대한 예측값
        :rtype: Tensor
        """
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    

def create_model(num_classes):
    model = NeuralNetwork(num_classes=num_classes)
    return model.cuda()