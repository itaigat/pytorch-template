"""
Modified types
"""

from torch import Tensor
from pathlib import Path
from typing import Union, List, Tuple, Dict

PathT = Union[str, Path]
Metrics = Dict[str, float]
Scores = Tuple[float, float]
InputSample = Union[Tensor, List[Tensor]]
