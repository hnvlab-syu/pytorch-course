from torch import nn, Tensor

class NeuralNetwork(nn.Module):
    """학습과 추론에 사용되는 간단한 뉴럴 네트워크"""

    # 모델 초기화
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),            
        )

    # 모델 생성
    def forward(self, x: Tensor) -> Tensor:
        """순전파 진행하는 함수
        
        param x: 입력 이미지
        type x: Tensor
        return: 입력 이미지에 대한 예측값
        rtype: Tensor
        """

        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits