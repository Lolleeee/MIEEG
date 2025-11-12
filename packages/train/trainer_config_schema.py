import logging
from pydantic import BaseModel, Field, field_validator, ConfigDict, model_validator
from torch.utils.data import DataLoader
from torch import nn, optim, cuda
from torch import device as torchdevice
from typing import Optional, Literal, List, Any, Union
from packages.train.metrics import TorchMetric, RMSE, MSE, MAE, AxisCorrelation
from packages.train.loss import TorchLoss, TorchMSELoss, TorchL1Loss, SequenceVQVAELoss, VQAE23Loss
from packages.models.vqae_skip import SequenceVQAE as SequenceVQAE_Skip
from packages.models.vqae import VQVAE
from packages.models.vqae_23 import VQAE as VQAE23
from packages.models.test_models import SimpleVQVAE, SimpleAutoencoder
from packages.data_objects.dataset import TorchDataset, TestTorchDataset
from enum import Enum
import json




# ===== String-based Enums =====

class ModelType(str, Enum):
    SEQUENCE_VQAE_SKIP = "sequence_vqae_skip"
    VQVAE = "vqvae"
    #Test models
    SIMPLE_VQVAE = "simple_vqvae"
    SIMPLE_AE = "simple_ae"
    VQAE23 = "vqae23"

class OptimizerType(str, Enum):
    ADAM = "adam"
    ADAMW = "adamw"


class LossType(str, Enum):
    MSE = "mse"
    L1 = "l1"
    SEQVQVAELOSS = "seq_vqvae_loss"
    VQAE23LOSS = "vqae23_loss"


class DatasetType(str, Enum):
    """
     Dataset types available for training.
    Options:
    - TORCH_DATASET: Standard TorchDataset class.
    - TEST_TORCH_DATASET: A simple test dataset for unit testing.
    """
    TORCH_DATASET = "torch_dataset"
    TEST_TORCH_DATASET = "test_torch_dataset"


class MetricType(str, Enum):
    MAE = "mae"
    RMSE = "rmse"
    AxisCorrelation = "axis_correlation"
    MSE = "mse"


# ===== Mapping Dictionaries =====

MODEL_MAP = {
    ModelType.VQVAE: VQVAE,
    ModelType.SEQUENCE_VQAE_SKIP: SequenceVQAE_Skip,
    ModelType.SIMPLE_VQVAE: SimpleVQVAE,
    ModelType.SIMPLE_AE: SimpleAutoencoder,
    ModelType.VQAE23: VQAE23,
}

OPTIMIZER_MAP = {
    OptimizerType.ADAM: optim.Adam,
    OptimizerType.ADAMW: optim.AdamW,
}

LOSS_MAP = {
    LossType.MSE: TorchMSELoss,
    LossType.L1: TorchL1Loss,
    LossType.SEQVQVAELOSS: SequenceVQVAELoss,
    LossType.VQAE23LOSS: VQAE23Loss,
}

DATASET_MAP = {
    DatasetType.TORCH_DATASET: TorchDataset,
    DatasetType.TEST_TORCH_DATASET: TestTorchDataset,
}

METRIC_MAP = {
    MetricType.MAE: MAE,
    MetricType.RMSE: RMSE,
    MetricType.AxisCorrelation: AxisCorrelation,
    MetricType.MSE: MSE,
}


# ===== Config Classes =====

class SetSizes(BaseModel):
    train: float = Field(default=0.6, ge=0.0, le=1.0)
    val: float = Field(default=0.2, ge=0.0, le=1.0)
    test: float = Field(default=0.2, ge=0.0, le=1.0)

    @model_validator(mode='after')
    def validate_sum(self):
        total = self.train + self.val + self.test
        if not (0.99 <= total <= 1.01):  # Allow small floating point error
            raise ValueError(f"Split sizes must sum to 1.0, got {total}")
        return self


class DataLoaderConfig(BaseModel):
    set_sizes: SetSizes = Field(default_factory=SetSizes, description="Proportions for train/val/test splits")
    batch_size: int = Field(default=32, gt=0)
    norm_axes: Optional[List[int]] = Field(default=None, description="Axes to normalize model input over")
    target_norm_axes: Optional[List[int]] = Field(default=None, description="Axes to normalize target over")

class DatasetConfig(BaseModel):
    dataset_type: DatasetType = Field(default=DatasetType.TORCH_DATASET, description="Type of dataset to use")
    dataset_args: dict = Field(default_factory=dict, description="Arguments to instantiate the dataset")
    data_loader: DataLoaderConfig = Field(default_factory=DataLoaderConfig)

    # Properties to get actual classes
    @property
    def get_dataset_class(self):
        """Returns the actual dataset class"""
        return DATASET_MAP[self.dataset_type]


class Optimizer(BaseModel):
    type: OptimizerType = Field(default=OptimizerType.ADAM, description="Type of optimizer to use")
    lr: float = Field(default=1e-3, gt=0.0, description="Learning rate")
    asym_lr: Optional[List[dict]] = Field(default=None, description="Asymmetric learning rates (will be fed to optimizer as parameter groups)")
    weight_decay: float = Field(default=1e-4, ge=0.0, description="Weight decay")

    @field_validator('asym_lr')
    @classmethod
    def validate_asym_lr(cls, v):
        if v is None:
            return v
        if isinstance(v, list):
            for item in v:
                if 'params' not in item or 'lr' not in item:
                    raise ValueError("Each item in asym_lr must contain 'params' and 'lr' keys")
            return v
        raise TypeError("asym_lr must be a list of dicts")

    @property
    def get_optimizer_class(self):
        """Returns the actual optimizer class"""
        return OPTIMIZER_MAP[self.type]


class TrainLoop(BaseModel):
    epochs: int = Field(default=50, gt=0)
    runtime_validation: Optional[bool] = Field(default=False, description="Whether to perform validation during loops")

class PlotType(str, Enum):
    """Plotting states for training history visualization.
    Options:
    - TIGHT: Plot metrics and loss in the same figure.
    - EXTENDED: Plot metrics and loss in separate figures.
    - OFF: Disable plotting.
    """
    TIGHT = 'tight'
    EXTENDED = 'extended'
    OFF = 'off'


class HistoryPlotSchema(BaseModel):
    plot_type: PlotType = Field(default=PlotType.TIGHT, description="Type of plot for training history")
    save_path: Optional[str] = Field(default='./training_history', description="Path to save the training history plots")

class EarlyStoppingSchema(BaseModel):
    patience: int = Field(default=30, gt=0, description="Number of epochs with no improvement after which training will be stopped")
    min_delta: float = Field(default=0.0, ge=0.0, description="Minimum change to qualify as an improvement")
    metric: str = Field(default='loss', description="Metric to monitor for early stopping")
    mode: Literal["min", "max"] = Field(default="min")

class BackupManagerSchema(BaseModel):
    backup_interval: int = Field(default=10, gt=0, description="Interval (in epochs) to save model backups")
    backup_path: str = Field(default="./model_backups", description="Path to save model backups")
    metric: str = Field(default='loss', description="Metric to monitor for backup")
    mode: Literal["min", "max"] = Field(default="min", description="Mode for monitoring metric")

class ReduceLROnPlateauSchema(BaseModel):
    mode: Literal["min", "max"] = Field(default="min", description="Mode for monitoring metric")
    patience: int = Field(default=20, gt=0, description="Number of epochs with no improvement after which learning rate will be reduced")
    factor: float = Field(default=0.1, gt=0.0, lt=1.0, description="Factor by which the learning rate will be reduced")

class GradientLoggerSchema(BaseModel):
    interval: Optional[int] = Field(default=None, gt=0, description="Interval (in epochs) to log gradient norms. If None, logging is disabled.")

# TODO Add Model output keys validation maybe?
class Helpers(BaseModel):
    history_plot: Optional[HistoryPlotSchema] = Field(default_factory=HistoryPlotSchema)
    early_stopping: Optional[EarlyStoppingSchema] = Field(default_factory=EarlyStoppingSchema)
    backup_manager: Optional[BackupManagerSchema] = Field(default_factory=BackupManagerSchema)
    reduce_lr_on_plateau: Optional[ReduceLROnPlateauSchema] = Field(default_factory=ReduceLROnPlateauSchema)
    gradient_logger: Optional[GradientLoggerSchema] = Field(default_factory=GradientLoggerSchema)

class GradientControl(BaseModel):
    grad_clip: Optional[float] = Field(default=1.0, gt=0.0, description="Maximum gradient norm for clipping. If None, no clipping is applied.")
    use_amp: bool = Field(default=False, description="Whether to use Automatic Mixed Precision (AMP) for training.")

class Info(BaseModel):
    metrics: List[MetricType] = Field(default_factory=list, description="List of metrics to compute during training and validation")
    metrics_args: Optional[List[dict]] = Field(default=None, description="List of dicts with args to instantiate each metric")
    @property
    def get_metric_classes(self):
        """Returns list of actual metric classes"""
        return [METRIC_MAP.get(m) for m in self.metrics]
    
class ModelConfig(BaseModel):
    model_type: ModelType = Field(default=ModelType.VQVAE, description="Fully qualified model class name (e.g., 'models.MyModel')")
    model_kwargs: dict = Field(default_factory=dict, description="Kwargs to instantiate model")

class LossConfig(BaseModel):
    loss_type: LossType = Field(default=LossType.MSE, description="Type of loss function to use")
    loss_kwargs: dict = Field(default_factory=dict, description="Kwargs to instantiate loss function")

class SanityCheckConfig(BaseModel):
    set_sizes: SetSizes = Field(default_factory=SetSizes)
    epochs: int = Field(default=5, gt=0, description="Number of epochs to run sanity check")
    enabled: bool = Field(default=False, description="Whether to enable sanity check")
    nsamples: int = Field(default=10, gt=0, description="Number of samples to use in TestTorchDataset for sanity check")
    shape: tuple = Field(default=(25, 7, 5, 250), description="Shape of samples in TestTorchDataset for sanity check")
    
class TrainerConfig(BaseModel):
    """
    Main training configuration. 
    Note: model instance is NOT included in JSON serialization
    """
    model: ModelConfig = Field(default_factory=ModelConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    optimizer: Optimizer = Field(default_factory=Optimizer)
    gradient_control: GradientControl = Field(default_factory=GradientControl)
    train_loop: TrainLoop = Field(default_factory=TrainLoop)
    helpers: Helpers = Field(default_factory=Helpers)
    info: Info = Field(default_factory=Info)
    device: str = Field(
        default_factory=lambda: "cuda" if cuda.is_available() else "cpu"
    )
    sanity_check: Optional[SanityCheckConfig] = Field(default_factory=SanityCheckConfig)
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility. If None, no seed is set.")


    @property
    def get_loss_class(self):
        """Returns the actual loss class"""
        return LOSS_MAP[self.loss.loss_type]

    @property
    def get_device(self):
        """Returns torch device"""
        return torchdevice(self.device)

    @property
    def get_model_class(self):
        """Returns the actual model class"""
        return MODEL_MAP[self.model.model_type]

# ===== Usage Functions =====

def save_config(config: TrainerConfig, path: str):
    """Save config to JSON"""
    with open(path, 'w') as f:
        json.dump(config.model_dump(), f, indent=2)
    logging.info(f"Config saved to {path}")


def load_config(path: str) -> TrainerConfig:
    """Load config from JSON"""
    with open(path, 'r') as f:
        config_dict = json.load(f)
    return TrainerConfig(**config_dict)
    