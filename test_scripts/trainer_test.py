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
from packages.models.vqae_light import VQAELight, VQAELightConfig

@dataclass

@dataclass
class VQAELightConfig:
    """Configuration for the VQ-VAE model."""
    use_quantizer: bool = False
    
    # Data shape parameters
    num_input_channels: int = 2   # Power + Phase
    num_freq_bands: int = 30
    spatial_rows: int = 7
    spatial_cols: int = 5
    time_samples: int = 80        # Fixed time window size
    orig_channels: int = 32       # Target output channels
    
    # Encoder parameters
    encoder_2d_channels: list = None   # [16, 32]
    encoder_3d_channels: list = None   # [32, 64]
    embedding_dim: int = 64
    
    # VQ parameters
    codebook_size: int = 256
    commitment_cost: float = 0.25
    ema_decay: float = 0.99
    epsilon: float = 1e-5
    
    # Decoder parameters
    decoder_channels: list = None      # [64, 32]
    
    # Dropout
    dropout_2d: float = 0.05
    dropout_3d: float = 0.05
    dropout_bottleneck: float = 0.1
    dropout_decoder: float = 0.05
    
    # Architecture
    use_separable_conv: bool = True
    use_group_norm: bool = True
    num_groups: int = 8
    use_residual: bool = True
    use_squeeze_excitation: bool = True


    

config = {
    'model': {
        'model_type': ModelType.COMVQAE23,
        'model_kwargs': {
            'config': VQAELightConfig()
        }
    },
    'dataset': {
        'dataset_type': DatasetType.H5_DATASET,
        'dataset_args': {
            'h5_path': 'scripts/test_output/TEST/motor_eeg_dataset.h5',
        },
        'data_loader': {
            'set_sizes': {
                'train': 0.6,
                'val': 0.2,
                'test': 0.2
            },
            'batch_size': 32,
            'norm_axes': (0, 5),
            'target_norm_axes': (0, 2)
        }
    },
    'loss': {
        'loss_type': LossType.VQAE23LOSS,
        'loss_kwargs': {}
    },
    'optimizer': {
        'type': OptimizerType.ADAMW,
        'lr': 0.001,
        'asym_lr': None,
        'weight_decay': 0.0001
    },
    'gradient_control': {
        'grad_clip': None,
        'use_amp': False
    },
    'train_loop': {
        'epochs': 100,
        'runtime_validation': False
    },
    'helpers': {
        'history_plot': {
            'plot_type': PlotType.EXTENDED,
            'save_path': './training_history',
            'metrics_logged': ['MSE']
        },
        'early_stopping': {
            'patience': 1000,
            'min_delta': 0.0,
            'metric': 'loss',
            'mode': 'min'
        },
        'backup_manager': None,
        'reduce_lr_on_plateau': None,
        'gradient_logger': None, #{'interval': 1}
        'custom_plotter': {
            'plot_function': CustomPlotTypes.RECONSTRUCTIONS,
            'plot_function_args': {},
            'plot_interval': 1000
        }
    },
    'info': {
        'metrics': [MetricType.MSE],
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