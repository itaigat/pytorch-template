"""
Main utils file, all utils functions that are not related to train.
"""

import os

import hydra
import torch
import schema
import operator
import functools

from torch import nn
from typing import Dict
from utils.types import PathT
from collections import MutableMapping
from utils.config_schema import CFG_SCHEMA
from omegaconf import DictConfig, OmegaConf


def get_model_string(model: nn.Module) -> str:
    """
    This function returns a string representing a model (all layers and parameters).
    :param model: instance of a model
    :return: model \n parameters
    """
    model_string: str = str(model)

    n_params = 0
    for w in model.parameters():
        n_params += functools.reduce(operator.mul, w.size(), 1)

    model_string += '\n'
    model_string += f'Params: {n_params}'

    return model_string


def set_seed(seed: int) -> None:
    """
    Sets a seed
    :param seed: seed to set
    """
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dir(path: PathT) -> None:
    """
    Given a path, creating a directory in it
    :param path: string of the path
    """
    if not os.path.exists(path):
        os.mkdir(path)


def warning_print(text: str) -> None:
    """
    This function prints text in yellow to indicate warning
    :param text: text to be printed
    """
    print(f'\033[93m{text}\033[0m')


def validate_input(cfg: DictConfig) -> None:
    """
    Validate the configuration file against schema.
    :param cfg: configuration file to validate
    """
    cfg_types = schema.Schema(CFG_SCHEMA)
    cfg_types.validate(OmegaConf.to_container(cfg))


def _flatten_dict(d: MutableMapping, parent_key: str = '', sep: str = '_') -> Dict:
    """
    Flatten a dictionary.
    For example:
    {'a': 1, 'c': {'a': 2, 'b': {'x': 5, 'y' : 10}}, 'd': [1, 2, 3]} ->
    {'a': 1, 'c_a': 2, 'c_b_x': 5, 'd': [1, 2, 3], 'c_b_y': 10}
    :param d: dictionary to flat
    :param parent_key: key to start from
    :param sep: separator symbol
    :return: flatten dictionary
    """
    items = []

    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))

    return dict(items)


def get_flatten_dict(cfg: DictConfig) -> Dict:
    """
    Returns flatten dictionary, given a config dictionary
    :param cfg: config file
    :return: flatten dictionary
    """
    return _flatten_dict(cfg)


def init(cfg: DictConfig) -> None:
    """
    :cfg: hydra configuration file
    """
    # TODO: Trains
    os.chdir(hydra.utils.get_original_cwd())
    validate_input(cfg)
