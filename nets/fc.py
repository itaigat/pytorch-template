"""
Fully connected network for example.
"""

from abc import ABCMeta
from torch import Tensor, nn
from torch.nn.utils.weight_norm import weight_norm


class FCNet(nn.Module, metaclass=ABCMeta):
    """
    Fully connected network for example.
    Usage example:
    FCNet([4096, 1000, 200], 'ReLU', 0.2) - for a fully connected with three layers, ReLU, weight normalization and
    dropout of 0.2.
    """
    def __init__(self, dims, activation: str = 'ReLU', dropout: float = 0.0) -> None:
        super(FCNet, self).__init__()
        layers = []

        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]

            if 0 < dropout:
                layers.append(nn.Dropout(dropout))

            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))

            if activation != '':
                layers.append(getattr(nn, activation)())

        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))

        if activation != '':
            layers.append(getattr(nn, activation)())

        self.seq = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward x through the network
        :param x:
        :return: logits
        """
        return self.seq(x)
