"""
    Example for a simple model
"""

from abc import ABCMeta
from nets.fc import FCNet
from torch import nn, Tensor


class MyModel(nn.Module, metaclass=ABCMeta):
    """
    Example for a simple model
    """
    def __init__(self, input_dim: int = 50, num_hid: int = 256, output_dim: int = 2, dropout: float = 0.2):
        super(MyModel, self).__init__()
        self.classifier = FCNet([input_dim, num_hid, num_hid, output_dim], 'ReLU', dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward x through MyModel
        :param x:
        :return:
        """
        return self.classifier(x)
