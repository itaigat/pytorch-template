"""
Train logger class
"""

import os
import torch
import logging

from torch import nn
from typing import Dict
from utils import main_utils
from datetime import datetime
from torch.optim import Optimizer
from utils.types import PathT, InputSample
from torch.utils.tensorboard import SummaryWriter


class TrainLogger:
    """
    Train logger class, covers all reports to tensorboard, console and files.
    """
    def __init__(self, exp_name_prefix: str = '', logs_dir: PathT = 'logs'):
        """
        :param exp_name_prefix: const prefix for specific experiment. For example, Random_exp_..
        :param logs_dir: directory for text file and tensorboard files
        """
        self.exp_name_prefix = exp_name_prefix

        main_utils.make_dir(logs_dir)

        self.exp_name = f'{exp_name_prefix}_{self._get_time_string()}'
        self.exp_dir = os.path.join(logs_dir, self.exp_name)

        # Init tensorboard
        tensorboard_path = os.path.join(logs_dir, 'tensorboard', self.exp_name)
        self.tensorboard_writer = SummaryWriter(tensorboard_path)

        # Init console and file logger
        self.logger = self._init_logger(self.exp_dir)

    def write(self, text: str, epoch: int = None, severity: str = 'info') -> None:
        """
        Output log in severity of info or warning. Specify epoch if given
        :param text: text of the log
        :param epoch: if given, adds the current epoch.
        :param severity: info or warning are valid
        """
        log = ''

        if epoch is not None:
            log += f'(EPOCH {epoch}) '
        log += f'{text}'

        if severity == 'warning':
            self.logger.warning(log)
        else:
            self.logger.info(log)

    def report_metrics_hyper_params(self, hyper_params: Dict, metrics: Dict) -> None:
        """
        Adds hyper parameters and metrics to tensorboard
        :param hyper_params: dictionary holds all the hyper parameters {hyper_param: value}
        :param metrics: dictionary holds all the metrics {metric: value}
        """
        self.tensorboard_writer.add_hparams(hyper_params, metrics)

    def report_scalar(self, tag: str, scalar_value: float, step: int) -> None:
        """
        Report a scalar to tensorboard
        :param tag: report the scalar under tag
        :param scalar_value:
        :param step: epoch
        """
        self.tensorboard_writer.add_scalar(tag, scalar_value, step)

    def report_graph(self, model: nn.Module, model_input: InputSample) -> None:
        """
        Report a model structure to tensorboard
        :param model: a model instance
        :param model_input: tensor or list of tensors
        """
        self.tensorboard_writer.add_graph(model, model_input)

    def save_model(self, model: nn.Module, epoch: int, optimizer: Optimizer = None) -> None:
        """
        Saving a model
        :param model: a model instance with trained weights
        :param epoch: current epoch
        :param optimizer: (optional), add saves the optimizer state
        """
        model_dict = {
            'epoch': epoch,
            'model_state': model.state_dict()
        }

        if optimizer is not None:
            model_dict['optimizer_state'] = optimizer.state_dict()

        model_path = os.path.join(self.exp_dir, 'model.pth')

        torch.save(model_dict, model_path)

    @staticmethod
    def _get_time_string() -> str:
        """
        :return: string with the current time in the format 'month_day_hour_minute_second'
        """
        time = datetime.now()

        return f'{time.month}_{time.day}_{time.hour}_{time.minute}_{time.second}'

    @staticmethod
    def _init_logger(exp_dir: PathT) -> logging.Logger:
        """
        Create a logger instance
        :param exp_dir: the directory that will hold the log file
        :return: a logger instance
        """
        # Create logger
        _logger = logging.getLogger('TRAIN')
        _logger.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter('[%(asctime)s] [%(name)s] - %(message)s')

        # Add console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        _logger.addHandler(ch)

        # Add file handler
        fh = logging.FileHandler(filename=f'{exp_dir}.log')
        fh.setFormatter(formatter)
        _logger.addHandler(fh)

        return _logger

    def report_scalars(self, scalars, epoch):
        """
        Report batch of scalars
        :param scalars: {scalar_key: scalar_value}. For instance: {'Accuracy_train': 99.32}
        :param epoch:
        """
        for scalar, scalar_value in scalars.items():
            self.report_scalar(scalar, scalar_value, epoch)

    def write_epoch_statistics(self, epoch: int, epoch_time: float, train_loss: float, norm: float,
                               train_score: float, eval_score: float) -> None:
        """
        Write multiple metrics
        :param epoch:
        :param epoch_time:
        :param train_loss:
        :param norm:
        :param train_score:
        :param eval_score:
        """
        text = 'epoch %d, time: %.2f' % (epoch, epoch_time)
        text += '\n'
        text += 'train_loss: %.2f, norm: %.4f, score: %.2f' % (train_loss, norm, train_score)
        text += '\n'
        text += 'eval score: %.2f' % eval_score

        self.write(text, epoch)


if __name__ == '__main__':
    logger = TrainLogger(exp_name_prefix='test', logs_dir='test_dir')
    for i in range(15):
        logger.write('I am without epoch')
    for i in range(15):
        logger.write('I am with epoch', epoch=i)
