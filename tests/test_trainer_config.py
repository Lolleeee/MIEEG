import json
import tempfile
import pytest
import torch
from pydantic import ValidationError

from packages.train.trainer_config_schema import (
    TrainerConfig,
    ModelConfig,
    DatasetConfig,
    LossConfig,
    Optimizer,
    SetSizes,
    DataLoaderConfig,
    Info,
    Helpers,
    TrainLoop,
    GradientControl,
    EarlyStoppingSchema,
    BackupManagerSchema,
    ReduceLROnPlateauSchema,
    GradientLoggerSchema,
    HistoryPlot,
    PlotStates,
    ModelType,
    OptimizerType,
    LossType,
    DatasetType,
    MetricType,
    save_config,
    load_config,
    MODEL_MAP,
    OPTIMIZER_MAP,
    LOSS_MAP,
    DATASET_MAP,
    METRIC_MAP,
)


# ===== SetSizes Tests =====

def test_setsizes_default_values():
    """Test SetSizes creates with default split values"""
    sizes = SetSizes()
    assert sizes.train == 0.6
    assert sizes.val == 0.2
    assert sizes.test == 0.2


def test_setsizes_validation_accepts_valid_sum():
    """Test SetSizes accepts splits that sum to 1.0"""
    sizes = SetSizes(train=0.7, val=0.2, test=0.1)

    assert sizes.train + sizes.val + sizes.test == pytest.approx(1.0, rel=1e-6)


def test_setsizes_validation_rejects_invalid_sum():
    """Test SetSizes rejects splits that don't sum to 1.0"""
    with pytest.raises(ValidationError, match="Split sizes must sum to 1.0"):
        SetSizes(train=0.8, val=0.4, test=0.3)


def test_setsizes_validation_accepts_floating_point_tolerance():
    """Test SetSizes accepts small floating point errors"""
    sizes = SetSizes(train=0.6, val=0.2, test=0.20000001)  # Within tolerance
    assert 0.99 <= (sizes.train + sizes.val + sizes.test) <= 1.01


def test_setsizes_validation_rejects_out_of_range():
    """Test SetSizes rejects values outside [0, 1]"""
    with pytest.raises(ValidationError):
        SetSizes(train=1.5, val=0.2, test=0.2)
    
    with pytest.raises(ValidationError):
        SetSizes(train=-0.1, val=0.2, test=0.2)


# ===== DataLoaderConfig Tests =====

def test_dataloaderconfig_defaults():
    """Test DataLoaderConfig default values"""
    config = DataLoaderConfig()
    assert config.batch_size == 32
    assert config.norm_axes is None
    assert isinstance(config.set_sizes, SetSizes)


def test_dataloaderconfig_custom_values():
    """Test DataLoaderConfig with custom values"""
    config = DataLoaderConfig(
        batch_size=64,
        norm_axes=[0, 1, 2],
        set_sizes=SetSizes(train=0.8, val=0.1, test=0.1)
    )
    assert config.batch_size == 64
    assert config.norm_axes == [0, 1, 2]
    assert config.set_sizes.train == 0.8


def test_dataloaderconfig_rejects_invalid_batch_size():
    """Test DataLoaderConfig rejects non-positive batch size"""
    with pytest.raises(ValidationError):
        DataLoaderConfig(batch_size=0)
    
    with pytest.raises(ValidationError):
        DataLoaderConfig(batch_size=-5)


# ===== DatasetConfig Tests =====

def test_datasetconfig_defaults():
    """Test DatasetConfig default values"""
    config = DatasetConfig()
    assert config.dataset_type == DatasetType.TORCH_DATASET
    assert config.dataset_args == {}
    assert isinstance(config.data_loader, DataLoaderConfig)


def test_datasetconfig_get_dataset_class_torch_dataset():
    """Test get_dataset_class returns correct class for TORCH_DATASET"""
    config = DatasetConfig(dataset_type=DatasetType.TORCH_DATASET)
    cls = config.get_dataset_class
    assert cls == DATASET_MAP[DatasetType.TORCH_DATASET]


def test_datasetconfig_get_dataset_class_test_torch_dataset():
    """Test get_dataset_class returns correct class for TEST_TORCH_DATASET"""
    config = DatasetConfig(dataset_type=DatasetType.TEST_TORCH_DATASET)
    cls = config.get_dataset_class
    assert cls == DATASET_MAP[DatasetType.TEST_TORCH_DATASET]


def test_datasetconfig_with_custom_args():
    """Test DatasetConfig with custom dataset arguments"""
    config = DatasetConfig(
        dataset_type=DatasetType.TEST_TORCH_DATASET,
        dataset_args={"nsamples": 100, "shape": (10, 20)}
    )
    assert config.dataset_args["nsamples"] == 100
    assert config.dataset_args["shape"] == (10, 20)


# ===== Optimizer Tests =====

def test_optimizer_defaults():
    """Test Optimizer default values"""
    opt = Optimizer()
    assert opt.type == OptimizerType.ADAM
    assert opt.lr == 1e-3
    assert opt.asym_lr is None
    assert opt.weight_decay == 1e-4


def test_optimizer_custom_values():
    """Test Optimizer with custom values"""
    opt = Optimizer(
        type=OptimizerType.ADAMW,
        lr=5e-4,
        weight_decay=1e-5
    )
    assert opt.type == OptimizerType.ADAMW
    assert opt.lr == 5e-4
    assert opt.weight_decay == 1e-5


def test_optimizer_rejects_negative_lr():
    """Test Optimizer rejects negative learning rate"""
    with pytest.raises(ValidationError):
        Optimizer(lr=-1e-3)


def test_optimizer_rejects_zero_lr():
    """Test Optimizer rejects zero learning rate"""
    with pytest.raises(ValidationError):
        Optimizer(lr=0.0)


def test_optimizer_rejects_negative_weight_decay():
    """Test Optimizer rejects negative weight decay"""
    with pytest.raises(ValidationError):
        Optimizer(weight_decay=-1e-4)


def test_optimizer_asym_lr_valid():
    """Test Optimizer accepts valid asymmetric learning rate"""
    asym_lr = [
        {"params": [1, 2, 3], "lr": 1e-3},
        {"params": [4, 5], "lr": 1e-5}
    ]
    opt = Optimizer(asym_lr=asym_lr)
    assert len(opt.asym_lr) == 2
    assert opt.asym_lr[0]["lr"] == 1e-3


def test_optimizer_asym_lr_rejects_missing_params():
    """Test Optimizer rejects asym_lr without 'params' key"""
    with pytest.raises(ValueError, match="must contain 'params' and 'lr' keys"):
        Optimizer(asym_lr=[{"lr": 1e-3}])


def test_optimizer_asym_lr_rejects_missing_lr():
    """Test Optimizer rejects asym_lr without 'lr' key"""
    with pytest.raises(ValueError, match="must contain 'params' and 'lr' keys"):
        Optimizer(asym_lr=[{"params": [1, 2]}])


def test_optimizer_asym_lr_rejects_non_list():
    """Test Optimizer rejects non-list asym_lr"""
    with pytest.raises(ValidationError, match="Input should be a valid list"):
        Optimizer(asym_lr="not-a-list")


def test_optimizer_get_optimizer_class_adam():
    """Test get_optimizer_class returns Adam"""
    opt = Optimizer(type=OptimizerType.ADAM)
    assert opt.get_optimizer_class == torch.optim.Adam


def test_optimizer_get_optimizer_class_adamw():
    """Test get_optimizer_class returns AdamW"""
    opt = Optimizer(type=OptimizerType.ADAMW)
    assert opt.get_optimizer_class == torch.optim.AdamW


# ===== Helper Schema Tests =====

def test_earlystopping_defaults():
    """Test EarlyStoppingSchema default values"""
    es = EarlyStoppingSchema()
    assert es.patience == 30
    assert es.min_delta == 0.0


def test_earlystopping_rejects_zero_patience():
    """Test EarlyStoppingSchema rejects zero patience"""
    with pytest.raises(ValidationError):
        EarlyStoppingSchema(patience=0)


def test_earlystopping_rejects_negative_min_delta():
    """Test EarlyStoppingSchema rejects negative min_delta"""
    with pytest.raises(ValidationError):
        EarlyStoppingSchema(min_delta=-0.01)


def test_backupmanager_defaults():
    """Test BackupManagerSchema default values"""
    bm = BackupManagerSchema()
    assert bm.backup_interval == 10
    assert bm.backup_path == "./model_backups"


def test_reducelronplateau_defaults():
    """Test ReduceLROnPlateauSchema default values"""
    rlr = ReduceLROnPlateauSchema()
    assert rlr.mode == "min"
    assert rlr.patience == 20
    assert rlr.factor == 0.1


def test_reducelronplateau_rejects_invalid_mode():
    """Test ReduceLROnPlateauSchema rejects invalid mode"""
    with pytest.raises(ValidationError):
        ReduceLROnPlateauSchema(mode="invalid")


def test_reducelronplateau_rejects_factor_greater_than_one():
    """Test ReduceLROnPlateauSchema rejects factor >= 1.0"""
    with pytest.raises(ValidationError):
        ReduceLROnPlateauSchema(factor=1.0)
    
    with pytest.raises(ValidationError):
        ReduceLROnPlateauSchema(factor=1.5)


def test_gradientcontrolschema_defaults():
    """Test GradientControlSchema default values"""
    gc = GradientLoggerSchema()
    assert gc.interval is None


def test_helpers_defaults():
    """Test Helpers default values"""
    helpers = Helpers()
    assert isinstance(helpers.early_stopping, EarlyStoppingSchema)
    assert isinstance(helpers.backup_manager, BackupManagerSchema)
    assert isinstance(helpers.reduce_lr_on_plateau, ReduceLROnPlateauSchema)
    assert isinstance(helpers.gradient_logger, GradientLoggerSchema)


def test_helpers_with_none_values():
    """Test Helpers accepts None for optional fields"""
    helpers = Helpers(
        early_stopping=None,
        backup_manager=None,
        reduce_lr_on_plateau=None,
        gradient_logger=None
    )
    assert helpers.early_stopping is None
    assert helpers.backup_manager is None


# ===== GradientControl Tests =====

def test_gradientcontrol_defaults():
    """Test GradientControl default values"""
    gc = GradientControl()
    assert gc.grad_clip == 1.0
    assert gc.use_amp is False
    assert gc.grad_logging_interval is None


def test_gradientcontrol_rejects_zero_grad_clip():
    """Test GradientControl rejects zero grad_clip"""
    with pytest.raises(ValidationError):
        GradientControl(grad_clip=0.0)


def test_gradientcontrol_rejects_negative_grad_clip():
    """Test GradientControl rejects negative grad_clip"""
    with pytest.raises(ValidationError):
        GradientControl(grad_clip=-1.0)


def test_gradientcontrol_accepts_none_grad_clip():
    """Test GradientControl accepts None for grad_clip"""
    gc = GradientControl(grad_clip=None)
    assert gc.grad_clip is None


# ===== Info Tests =====

def test_info_defaults():
    """Test Info default values"""
    info = Info()
    assert isinstance(info.history_plot, HistoryPlot)
    assert info.metrics == []
    assert info.metrics_args is None


def test_info_get_metric_classes():
    """Test Info.get_metric_classes returns correct classes"""
    info = Info(metrics=[MetricType.MAE, MetricType.MSE, MetricType.RMSE])
    classes = info.get_metric_classes
    
    assert len(classes) == 3
    assert classes[0] == METRIC_MAP[MetricType.MAE]
    assert classes[1] == METRIC_MAP[MetricType.MSE]
    assert classes[2] == METRIC_MAP[MetricType.RMSE]


def test_info_with_metric_args():
    """Test Info with metric arguments"""
    info = Info(
        metrics=[MetricType.MAE],
        metrics_args=[{"input_is_target": True}]
    )
    assert len(info.metrics_args) == 1
    assert info.metrics_args[0]["input_is_target"] is True


# ===== HistoryPlot Tests =====

def test_historyplot_defaults():
    """Test HistoryPlot default values"""
    hp = HistoryPlot()
    assert hp.state == PlotStates.TIGHT
    assert hp.save_path == './training_history'
    assert hp.plot_composite_loss is False


def test_historyplot_custom_values():
    """Test HistoryPlot with custom values"""
    hp = HistoryPlot(
        state=PlotStates.EXTENDED,
        save_path="/custom/path",
        plot_composite_loss=True
    )
    assert hp.state == PlotStates.EXTENDED
    assert hp.save_path == "/custom/path"
    assert hp.plot_composite_loss is True


# ===== ModelConfig Tests =====

def test_modelconfig_with_values():
    """Test ModelConfig with values"""
    mc = ModelConfig(
        model_type=ModelType.VQVAE,
        model_kwargs={"in_channels": 10}
    )
    assert mc.model_type == ModelType.VQVAE
    assert mc.model_kwargs["in_channels"] == 10


# ===== LossConfig Tests =====

def test_lossconfig_defaults():
    """Test LossConfig default values"""
    lc = LossConfig()
    assert lc.loss_type == LossType.MSE
    assert lc.loss_kwargs == {}


def test_lossconfig_custom_values():
    """Test LossConfig with custom values"""
    lc = LossConfig(
        loss_type=LossType.L1,
        loss_kwargs={"reduction": "mean"}
    )
    assert lc.loss_type == LossType.L1
    assert lc.loss_kwargs["reduction"] == "mean"


# ===== TrainLoop Tests =====

def test_trainloop_defaults():
    """Test TrainLoop default values"""
    tl = TrainLoop()
    assert tl.epochs == 50


def test_trainloop_rejects_zero_epochs():
    """Test TrainLoop rejects zero epochs"""
    with pytest.raises(ValidationError):
        TrainLoop(epochs=0)


def test_trainloop_rejects_negative_epochs():
    """Test TrainLoop rejects negative epochs"""
    with pytest.raises(ValidationError):
        TrainLoop(epochs=-10)


# ===== TrainerConfig Tests =====

def test_trainerconfig_minimal():
    """Test TrainerConfig with minimal valid configuration"""
    config = TrainerConfig(
        model=ModelConfig(
            model_type=ModelType.VQVAE,
            model_kwargs={"in_channels": 10, "hidden_dims": [16], 
                         "num_embeddings": 64, "embedding_dim": 16}
        )
    )
    assert isinstance(config.model, ModelConfig)
    assert isinstance(config.dataset, DatasetConfig)
    assert isinstance(config.loss, LossConfig)
    assert isinstance(config.optimizer, Optimizer)


def test_trainerconfig_get_model_class():
    """Test TrainerConfig.get_model_class property"""
    config = TrainerConfig(
        model=ModelConfig(
            model_type=ModelType.VQVAE,
            model_kwargs={}
        )
    )
    assert config.get_model_class == MODEL_MAP[ModelType.VQVAE]


def test_trainerconfig_get_loss_class():
    """Test TrainerConfig.get_loss_class property"""
    config = TrainerConfig(
        model=ModelConfig(model_type=ModelType.VQVAE, model_kwargs={}),
        loss=LossConfig(loss_type=LossType.L1)
    )
    assert config.get_loss_class == LOSS_MAP[LossType.L1]


def test_trainerconfig_get_device():
    """Test TrainerConfig.get_device property"""
    config = TrainerConfig(
        model=ModelConfig(model_type=ModelType.VQVAE, model_kwargs={}),
        device="cpu"
    )
    device = config.get_device
    assert isinstance(device, torch.device)
    assert device.type == "cpu"


def test_trainerconfig_device_auto_detect():
    """Test TrainerConfig auto-detects CUDA availability"""
    config = TrainerConfig(
        model=ModelConfig(model_type=ModelType.VQVAE, model_kwargs={})
    )
    # Device should be either 'cuda' or 'cpu' depending on availability
    assert config.device in ["cuda", "cpu"]


# ===== Serialization Tests =====

def test_save_and_load_config(tmp_path):
    """Test saving and loading TrainerConfig"""
    config = TrainerConfig(
        model=ModelConfig(
            model_type=ModelType.VQVAE,
            model_kwargs={"in_channels": 3, "hidden_dims": [16], 
                         "num_embeddings": 128, "embedding_dim": 32}
        ),
        dataset=DatasetConfig(
            dataset_type=DatasetType.TEST_TORCH_DATASET,
            dataset_args={"nsamples": 50, "shape": (3, 16)}
        ),
        loss=LossConfig(loss_type=LossType.MSE),
        optimizer=Optimizer(type=OptimizerType.ADAM, lr=1e-4),
        device="cpu",
    )

    # Save config
    config_path = tmp_path / "config.json"
    save_config(config, str(config_path))

    # Load config
    loaded = load_config(str(config_path))

    # Verify fields preserved
    assert loaded.device == "cpu"
    assert loaded.model.model_type == ModelType.VQVAE
    assert loaded.model.model_kwargs["in_channels"] == 3
    assert loaded.dataset.dataset_type == DatasetType.TEST_TORCH_DATASET
    assert loaded.optimizer.type == OptimizerType.ADAM
    assert loaded.optimizer.lr == 1e-4
    assert loaded.loss.loss_type == LossType.MSE


def test_config_model_dump():
    """Test TrainerConfig.model_dump() produces valid dict"""
    config = TrainerConfig(
        model=ModelConfig(
            model_type=ModelType.VQVAE,
            model_kwargs={"in_channels": 5}
        )
    )
    
    dumped = config.model_dump()
    
    assert isinstance(dumped, dict)
    assert "model" in dumped
    assert "dataset" in dumped
    assert "loss" in dumped
    assert dumped["model"]["model_type"] == "vqvae"


def test_config_roundtrip_with_all_fields(tmp_path):
    """Test full config roundtrip with all optional fields"""
    config = TrainerConfig(
        model=ModelConfig(
            model_type=ModelType.VQVAESKIP,
            model_kwargs={"in_channels": 7, "hidden_dims": [32, 64]}
        ),
        dataset=DatasetConfig(
            dataset_type=DatasetType.TORCH_DATASET,
            dataset_args={"root_folder": "/data"},
            data_loader=DataLoaderConfig(
                batch_size=64,
                norm_axes=[0, 2, 3],
                set_sizes=SetSizes(train=0.7, val=0.15, test=0.15)
            )
        ),
        loss=LossConfig(loss_type=LossType.L1, loss_kwargs={"reduction": "sum"}),
        optimizer=Optimizer(
            type=OptimizerType.ADAMW,
            lr=5e-4,
            weight_decay=1e-5
        ),
        gradient_control=GradientControl(
            grad_clip=0.5,
            use_amp=True,
            grad_logging_interval=50
        ),
        train_loop=TrainLoop(epochs=100),
        helpers=Helpers(
            early_stopping=EarlyStoppingSchema(patience=20, min_delta=0.001),
            backup_manager=BackupManagerSchema(backup_interval=5, backup_path="/backups"),
            reduce_lr_on_plateau=ReduceLROnPlateauSchema(mode="max", patience=15, factor=0.2)
        ),
        info=Info(
            metrics=[MetricType.MAE, MetricType.RMSE],
            history_plot=HistoryPlot(state=PlotStates.EXTENDED, plot_composite_loss=True)
        ),
        device="cuda"
    )

    path = tmp_path / "full_config.json"
    save_config(config, str(path))
    loaded = load_config(str(path))

    # Verify all fields
    assert loaded.model.model_type == ModelType.VQVAESKIP
    assert loaded.dataset.data_loader.batch_size == 64
    assert loaded.loss.loss_type == LossType.L1
    assert loaded.optimizer.lr == 5e-4
    assert loaded.gradient_control.use_amp is True
    assert loaded.train_loop.epochs == 100
    assert loaded.helpers.early_stopping.patience == 20
    assert loaded.info.metrics == [MetricType.MAE, MetricType.RMSE]
    assert loaded.device == "cuda"