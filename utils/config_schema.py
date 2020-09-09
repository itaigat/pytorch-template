"""
Schema for config file
"""

CFG_SCHEMA = {
    'main': {
        'experiment_name_prefix': str,
        'seed': int,
        'num_workers': int,
        'parallel': bool,
        'gpus_to_use': str,
        'trains': bool,
        'paths': {
            'train': str,
            'validation': str,
            'logs': str,
        },
    },
    'train': {
        'num_epochs': int,
        'grad_clip': float,
        'dropout': float,
        'num_hid': int,
        'batch_size': int,
        'save_model': bool,
        'lr': {
            'lr_value': float,
            'lr_decay': int,
            'lr_gamma': float,
            'lr_step_size': int,
        },
    },
}
