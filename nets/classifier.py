"""
Classifier for example.
"""


from abc import ABCMeta
from torch import Tensor, nn
from torch.nn.utils.weight_norm import weight_norm


class Classifier(nn.Module, metaclass=ABCMeta):
    """
    Classifier for example.
    Usage example:
    Classifier(1024, 512, 2, 0.2) - for a fully with weight normalization and dropout of 0.2. Out dimension of 0.2.
    """
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, dropout: float):
        super(Classifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]

        self.classifier = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward x through the classifier
        :param x:
        :return: logits
        """
        return self.classifier(x)
