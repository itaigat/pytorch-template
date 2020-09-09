"""
    Example data generation
"""

import torch
import pickle


train = {}
validation = {}


for i in range(1000):
    input_env_0 = torch.randn((50, )) * 0.5
    train[i] = {'label': 0, 'input': input_env_0}

    input_env_0 = torch.randn((50,)) * 0.5
    validation[i] = {'label': 0, 'input': input_env_0}


for i in range(1000, 2000):
    input_env_1 = torch.randn((50,)) * 0.2
    train[i] = {'label': 1, 'input': input_env_1}

    input_env_1 = torch.randn((50,)) * 0.2
    validation[i] = {'label': 1, 'input': input_env_1}


pickle.dump(train, open('../data/train.pkl', 'wb'))
pickle.dump(validation, open('../data/validation.pkl', 'wb'))
