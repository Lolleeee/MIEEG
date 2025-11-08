from pprint import pprint
import sys
from packages.train.trainer_config_schema import (
    PlotType,
    TrainerConfig,
    DatasetType,
    OptimizerType,
    LossType,
    ModelType,
    MetricType
)
from packages.train.training import Trainer

config = config = {
    'model': {
        'model_type': ModelType.SIMPLE_VQVAE,
        'model_kwargs': {
            'in_channels': 10,
            'embedding_dim': 16,
            'num_embeddings': 64
        }
    },
    'dataset': {
        'dataset_type': DatasetType.TEST_TORCH_DATASET,
        'dataset_args': {
            'nsamples': 48,
            'shape': (10,)
        },
        'data_loader': {
            'set_sizes': {
                'train': 0.6,
                'val': 0.2,
                'test': 0.2
            },
            'batch_size': 16,
            'norm_axes': None
        }
    },
    'loss': {
        'loss_type': LossType.MSE,
        'loss_kwargs': {}
    },
    'optimizer': {
        'type': OptimizerType.ADAM,
        'lr': 0.001,
        'asym_lr': None,
        'weight_decay': 0.0001
    },
    'gradient_control': {
        'grad_clip': None,
        'use_amp': False
    },
    'train_loop': {
        'epochs': 10,
        'runtime_validation': False
    },
    'helpers': {
        'history_plot': {
            'plot_type': PlotType.EXTENDED,
            'save_path': './training_history'
        },
        'early_stopping': {
            'patience': 2,
            'min_delta': 0.0,
            'metric': LossType.MSE,
            'mode': 'min'
        },
        'backup_manager': None,
        'reduce_lr_on_plateau': None,
        'gradient_logger': None
    },
    'info': {
        'metrics': [MetricType.MAE, MetricType.MSE],
        'metrics_args': None
    },
    'device': 'cpu',
    'sanity_check': {
        'set_sizes': {
            'train': 0.6,
            'val': 0.2,
            'test': 0.2
        },
        'epochs': 5,
        'enabled': True
    }
}


train_config =  TrainerConfig(**config)

trainer = Trainer(train_config.model_dump())
trainer.start()