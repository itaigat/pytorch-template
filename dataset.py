"""
Here, we create a custom dataset
"""
import torch
import pickle

from utils.types import PathT
from torch.utils.data import Dataset
from typing import Any, Tuple, Dict, List


class MyDataset(Dataset):
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

    def __getitem__(self, index: int) -> Tuple:
        return self.entries[index]['x'], self.entries[index]['y']

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
        with open(self.path, "rb") as features_file:
            features = pickle.load(features_file)

        return features

    def _get_entries(self) -> List:
        """
        This function create a list of all the entries. We will use it later in __getitem__
        :return: list of samples
        """
        entries = []

        for idx, item in self.features.items():
            entries.append(self._get_entry(item))

        return entries

    @staticmethod
    def _get_entry(item: Dict) -> Dict:
        """
        :item: item from the data. In this example, {'input': Tensor, 'y': int}
        """
        x = item['input']
        y = torch.Tensor([1, 0]) if item['label'] else torch.Tensor([0, 1])

        return {'x': x, 'y': y}
