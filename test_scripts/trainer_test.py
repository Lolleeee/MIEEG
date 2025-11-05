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

config = {
        "model": {
            "model_type": ModelType.SIMPLE_VQVAE,
            "model_kwargs": {
                "in_channels": 10,
                "embedding_dim": 16,
                "num_embeddings": 64
            }
        },
        "dataset": {
            "dataset_type": DatasetType.TEST_TORCH_DATASET,
            "dataset_args": {"nsamples": 48, "shape": (10,)},
            "data_loader": {"batch_size": 16}
        },
        "loss": {"loss_type": LossType.MSE},
        "optimizer": {"type": OptimizerType.ADAM, "lr": 1e-3},
        "train_loop": {"epochs": 10},
        "gradient_control": {"grad_clip": None, "use_amp": False},
        "info": {
            "metrics": [MetricType.MAE, MetricType.MSE],
        },
        "helpers": {
            "early_stopping": {"patience": 2, "min_delta": 0.0},
            "backup_manager": None,
            "reduce_lr_on_plateau": None,
            "gradient_logger": None,
            "history_plot": {"plot_type": PlotType.OFF}
        }
    }
train_config =  TrainerConfig(**config)

trainer = Trainer(train_config.model_dump())
trainer.start()

print(trainer.metrics_handler.metrics_eval)