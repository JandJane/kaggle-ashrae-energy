import pandas as pd
import numpy as np
import torch

random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)

device = 'cuda:0'

EPS = 10 ** -6

DIR = '/data1/trufanova/energy'


columns_config = {
    'numerical': ['square_feet', 'year_built', 'floor_count', 'air_temperature', 'cloud_coverage', 'dew_temperature',
                  'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed'
#                   , 'mean_target'
                 ],
    'categorical': [
#         'site_id', 'building_id', 
        'primary_use', 'wind_direction_cat', 'month', 'hour', 'season', 'daytime']
}

models_config = {
    'nn': {
        'init_config':
            {
                'net_config': {
                    'n_hidden': 1,
                    'batch_norm': False,
                    'dropout': False,
                    'k': 3
                },
                'lr': 0.001,
            },
        'train_config': {'n_epochs': 1},
    },
    'cb': {
        'init_config':
            {
                'model_config': {
                    'iterations': 100,
                    'learning_rate': 1,
                    'depth': 20,
                    'task_type': 'GPU',
                    'devices': '1',
                    'random_seed': 42,
                    'use_best_model': True,
                }
            },
        'train_config': {}
    }
}

