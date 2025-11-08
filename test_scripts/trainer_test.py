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
        'model_type': ModelType.SEQUENCE_VQAE_SKIP,
        'model_kwargs': {
            'chunk_shape': (25, 7, 5, 25),  # (C, H, W, T) per chunk
            'embedding_dim': 128,
            'codebook_size': 512,
            'num_downsample_stages': 3,
            'use_quantizer': False,
            'use_skip_connections': False,
            'skip_strength': 1.0,  
            'skip_strengths': None, 
            'skip_mode': 'concat',  
            'commitment_cost': 0.25,
            'decay': 0.99
        }
    },
    'dataset': {
        'dataset_type': DatasetType.TORCH_DATASET,
        'dataset_args': {
            'root_folder': 'scripts/test_output',
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
        'loss_type': LossType.SEQVQVAELOSS,
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
    },
    'seed': 42
}


train_config =  TrainerConfig(**config)

trainer = Trainer(train_config.model_dump())
trainer.start()