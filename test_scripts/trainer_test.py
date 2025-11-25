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
from packages.models.vqae_23 import VQVAEConfig

@dataclass
class Config(VQVAEConfig):
    """Configuration for the VQ-VAE model."""
    use_quantizer: bool = True  # Whether to use vector quantization
    # Data shape parameters
    num_freq_bands: int = 25          # F: Number of frequency bands
    spatial_rows: int = 7              # R: Spatial grid rows
    spatial_cols: int = 5              # C: Spatial grid cols
    time_samples: int = 250            # T: Time samples per clip
    chunk_dim: int = 50                # ChunkDim: Time chunk length
    orig_channels: int = 32            # Original EEG channels (R*C or separate)
    
    # Encoder parameters
    encoder_2d_channels: list = None   # [32, 64] - 2D conv channels
    encoder_3d_channels: list = None   # [64, 128, 256] - 3D conv channels
    embedding_dim: int = 32           # Final embedding dimension
    
    # VQ parameters
    codebook_size: int = 16           # Number of codebook vectors
    commitment_cost: float = 0.5      # Beta for commitment loss
    ema_decay: float = 0.9            # EMA decay for codebook updates
    epsilon: float = 1e-5              # Small constant for numerical stability
    
    # Decoder parameters
    decoder_channels: list = None

    dropout_2d: float = 0          # Dropout for 2D encoder
    dropout_3d: float = 0          # Dropout for 3D encoder
    dropout_bottleneck: float = 0  # Dropout at bottleneck
    dropout_decoder: float = 0     # Dropout for decoder
    

config = {
    'model': {
        'model_type': ModelType.VQAE23,
        'model_kwargs': {
            'config': Config()
        }
    },
    'dataset': {
        'dataset_type': DatasetType.H5_DATASET,
        'dataset_args': {
            'root_folder': '/media/lolly/SSD/WAYEEGGAL_dataset/0.69subset_250_eeg_wav',
            'nsamples': 25,
        },
        'data_loader': {
            'set_sizes': {
                'train': 0.6,
                'val': 0.2,
                'test': 0.2
            },
            'batch_size': 32,
            'norm_axes': (0, 4),
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
            'plot_interval': 9999
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