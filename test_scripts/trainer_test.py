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
    MetricType
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
    chunk_dim: int = 25                # ChunkDim: Time chunk length
    orig_channels: int = 32            # Original EEG channels (R*C or separate)
    
    # Encoder parameters
    encoder_2d_channels: list = None   # [32, 64] - 2D conv channels
    encoder_3d_channels: list = None   # [64, 128, 256] - 3D conv channels
    embedding_dim: int = 128           # Final embedding dimension
    
    # VQ parameters
    codebook_size: int = 512           # Number of codebook vectors
    commitment_cost: float = 0.25      # Beta for commitment loss
    ema_decay: float = 0.99            # EMA decay for codebook updates
    epsilon: float = 1e-5              # Small constant for numerical stability
    
    # Decoder parameters
    decoder_channels: list = None

config = {
    'model': {
        'model_type': ModelType.VQAE23,
        'model_kwargs': {
            'config': Config()
        }
    },
    'dataset': {
        'dataset_type': DatasetType.TORCH_DATASET,
        'dataset_args': {
            'root_folder': 'scripts/test_output/EEG+Wavelet',
        },
        'data_loader': {
            'set_sizes': {
                'train': 0.2,
                'val': 0.2,
                'test': 0.6
            },
            'batch_size': 16,
            'norm_axes': (0, 4),
            'target_norm_axes': (0, 2)
        }
    },
    'loss': {
        'loss_type': LossType.VQAE23LOSS,
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
        'epochs': 1000,
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
            'metric': 'loss',
            'mode': 'min'
        },
        'backup_manager': None,
        'reduce_lr_on_plateau': None,
        'gradient_logger': None#{'interval': 1}
    },
    'info': {
        'metrics': [MetricType.MAE, MetricType.MSE],
        'metrics_args': None
    },
    'device': 'cpu',
    'sanity_check': {
        'set_sizes': {
            'train': 0.2,
            'val': 0.2,
            'test': 0.6
        },
        'epochs': 10,
        'enabled': True,
        'nsamples': 10,
        'shape': (25, 7, 5, 250)  # (
    },
    'seed': 42
}


train_config =  TrainerConfig(**config)

trainer = Trainer(train_config.model_dump())
trainer.start()