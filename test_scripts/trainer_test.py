from pprint import pprint
from dataclasses import dataclass
import sys
from packages.train.trainer_config_schema import (
    PlotType,
    TrainerConfig,
    DatasetType,
    OptimizerType,
    LossType,
    ModelType,
    MetricType,
    CustomPlotTypes
)
from packages.train.training import Trainer
from packages.models.vqae_light import VQAELightConfig
from packages.data_objects.dataset import autoencoder_unpack_func
model_config = VQAELightConfig(
    use_quantizer=False,
    embedding_dim=64
)

    

config = {
    'model': {
        'model_type': ModelType.VQAE23_SMALL,
        'model_kwargs': {
            'config': model_config
        }
    },
    'dataset': {
        'dataset_type': DatasetType.H5_DATASET,
        'dataset_args': {
            'h5_path': '/media/lolly/SSD/motor_eeg_dataset/motor_eeg_dataset.h5',
            'unpack_func': autoencoder_unpack_func
        },
        'data_loader': {
            'set_sizes': {
                'train': 0.6,
                'val': 0.2,
                'test': 0.2
            },
            'batch_size': 32,
            'norm_axes': (0, 2),
            'target_norm_axes': (0, 2),
            'nsamples': 5
        }
    },
    'loss': {
        'loss_type': LossType.L1,
        'loss_kwargs': {}#{'power_weight': 0, 'magnitude_weight': 0, 'phase_weight': 0}
    },
    'optimizer': {
        'type': OptimizerType.ADAMW,
        'lr': 0.001,
        'asym_lr': None,
        'weight_decay': 0.0001
    },
    'gradient_control': {
        'grad_clip': 1.0,
        'use_amp': False
    },
    'train_loop': {
        'epochs': 200,
        'runtime_validation': False
    },
    'helpers': {
        'history_plot': {
            'plot_type': PlotType.EXTENDED,
            'save_path': './training_history',
            'metrics_logged': ['loss']
        },
        'early_stopping': {
            'patience': 1000,
            'min_delta': 0.0,
            'metric': 'loss',
            'mode': 'min'
        },
        'backup_manager': None,
        'reduce_lr_on_plateau': None,
        'gradient_logger': None, #{'interval': 1},
        'custom_plotter': {
            'plot_function': CustomPlotTypes.SCATTER,
            'plot_function_args': {},
            'plot_interval': 50
        }
    },
    'info': {
        'metrics': [],
        'metrics_args': None,

    },
    'device': 'cpu',
    'sanity_check': {
        'set_sizes': {
            'train': 0.2,
            'val': 0.2,
            'test': 0.6
        },
        'epochs': 10,
        'enabled': False,
        'nsamples': 10,
        'shape': (25, 7, 5, 250)  # (
    },
    'seed': 42
}


train_config =  TrainerConfig(**config)

trainer = Trainer(train_config.model_dump())
trainer.start()