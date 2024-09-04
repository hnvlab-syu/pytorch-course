from torch import nn, Tensor

class NeuralNetwork(nn.Module):
    """
    Simple neural network for training and inference

    Attributes:
        flatten (nn.Flatten): layer for flattening input images.
        linear_relu_stack (nn.Sequential): stack of layers for classification.
    """

    def __init__(self, num_classes: int):
        """
        Layers initialization to be used in the forward.

        Args:
            num_classes (int): number of output classes for classification.
        """
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward of the neural network.

        Args:
            x (Tensor): batch of input images.

        Returns:
            Tensor: outputs of the neural network for each class.
        """    

        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits