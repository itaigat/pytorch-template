"""
Here, we create a custom dataset
"""

from typing import Any, List
from utils.types import PathT
from torch.utils.data import Dataset


class MyDataset(Dataset):  # TODO: Add more documentation regards what to do in each function
    """
    Custom dataset template. Implement the empty functions.
    """
    def __init__(self, path: PathT) -> None:
        # Set variables
        self.path = path

        # Load features
        self.features = self._get_features()

        # Create list of entries
        self.entries = self._get_entries()

    def __getitem__(self, index: int) -> List:
        pass

    def __len__(self) -> int:
        """
        :return: the length of the dataset (number of sample).
        """
        return len(self.entries)

    def _get_features(self) -> Any:
        """
        Load all features into a structure (not necessarily dictionary). Think if you need/can load all the features
        into the memory.
        :return:
        :rtype:
        """
        features = {}
        path = self.path

        return features

    def _get_entries(self) -> List:
        """
        This function create a list of all the entries. We will use it later in __getitem__
        :return: list of samples
        """
        entries = []

        for item in self.features:
            pass

        return entries
