from packages.train.trainer_config_schema import (
    TrainerConfig,
    DatasetType,
    OptimizerType,
    LossType,
    ModelType,
    MetricType
)
from packages.train.training import Trainer

config = {
        "model": {
            "model_type": ModelType.SIMPLE_VQVAE,  # Changed from VQVAE
            "model_kwargs": {
                "in_channels": 10,
                "embedding_dim": 16
            }
        },
        "dataset": {
            "dataset_type": DatasetType.TEST_TORCH_DATASET,
            "dataset_args": {"nsamples": 32, "shape": (10,)},
            "data_loader": {"batch_size": 8}
        },
        "loss": {"loss_type": LossType.MSE},
        "optimizer": {"type": OptimizerType.ADAM, "lr": 1e-3},
        "train_loop": {"epochs": 2},
        "gradient_control": {"grad_clip": None, "use_amp": False},
    }
train_config =  TrainerConfig(**config)

trainer = Trainer(train_config.model_dump())
trainer.start()