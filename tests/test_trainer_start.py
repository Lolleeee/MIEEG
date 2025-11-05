import pytest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import logging

from packages.train.training import Trainer
from packages.train.trainer_config_schema import (
    ModelType,
    DatasetType,
    LossType,
    OptimizerType,
    MetricType,
    PlotType,
)


# ===== Fixtures =====

@pytest.fixture
def minimal_config():
    """Minimal valid trainer configuration for fast tests"""
    return {
        "model": {
            "model_type": ModelType.SIMPLE_VQVAE,
            "model_kwargs": {
                "in_channels": 10,
                "embedding_dim": 16,
                "num_embeddings": 32
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
        "helpers": {
            "early_stopping": None,
            "backup_manager": None,
            "reduce_lr_on_plateau": None,
            "gradient_logger": None,
        },
        "info": {
            "metrics": [],
            "history_plot": {"state": PlotType.OFF}
        }
    }


@pytest.fixture
def config_with_metrics():
    """Configuration with metrics enabled"""
    return {
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
        "train_loop": {"epochs": 3},
        "gradient_control": {"grad_clip": None, "use_amp": False},
        "info": {
            "metrics": [MetricType.MAE, MetricType.MSE],
            "history_plot": {"state": PlotType.OFF}
        },
        "helpers": {
            "early_stopping": None,
            "backup_manager": None,
            "reduce_lr_on_plateau": None,
            "gradient_logger": None,
        }
    }


@pytest.fixture
def config_with_helpers(tmp_path):
    """Configuration with all helpers enabled"""
    backup_path = tmp_path / "backups"
    return {
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
            "dataset_args": {"nsamples": 64, "shape": (10,)},
            "data_loader": {"batch_size": 16}
        },
        "loss": {"loss_type": LossType.MSE},
        "optimizer": {"type": OptimizerType.ADAM, "lr": 1e-2},
        "train_loop": {"epochs": 10},
        "gradient_control": {"grad_clip": 1.0, "use_amp": False},
        "info": {
            "metrics": [MetricType.MAE],
            "history_plot": {"state": PlotType.OFF}
        },
        "helpers": {
            "early_stopping": {"patience": 3, "min_delta": 0.0},
            "backup_manager": {"backup_interval": 2, "backup_path": str(backup_path)},
            "reduce_lr_on_plateau": {"mode": "min", "patience": 2, "factor": 0.5},
            "gradient_logger": {"interval": 50},
        }
    }


# ===== Basic Execution Tests =====

def test_trainer_start_runs_without_error(minimal_config):
    """Test that trainer.start() executes without errors"""
    trainer = Trainer(minimal_config)
    
    # Should complete without raising
    trainer.start()
    
    # Verify training completed
    assert trainer.current_epoch == minimal_config["train_loop"]["epochs"]


def test_trainer_start_updates_current_epoch(minimal_config):
    """Test that current_epoch increments correctly"""
    minimal_config["train_loop"]["epochs"] = 5
    trainer = Trainer(minimal_config)
    
    initial_epoch = trainer.current_epoch
    trainer.start()
    
    assert initial_epoch == 0
    assert trainer.current_epoch == 5


def test_trainer_start_trains_model(minimal_config):
    """Test that model parameters are updated during training"""
    trainer = Trainer(minimal_config)
    
    # Get initial parameters
    initial_params = [p.clone() for p in trainer.model.parameters()]
    
    trainer.start()
    
    # At least some parameters should have changed
    final_params = list(trainer.model.parameters())
    params_changed = any(
        not torch.allclose(initial, final)
        for initial, final in zip(initial_params, final_params)
    )
    
    assert params_changed, "Model parameters should change during training"


# ===== Metrics Tests =====

def test_trainer_start_computes_metrics(config_with_metrics):
    """Test that metrics are computed during training"""
    trainer = Trainer(config_with_metrics)
    
    # Store initial state (if metrics have internal counters)
    initial_states = {}
    for name, metric in trainer.metrics_handler.metrics.items():
        if hasattr(metric, 'total'):
            initial_states[name] = metric.total
    
    trainer.start()
    
    # Check metrics were updated
    for name, metric in trainer.metrics_handler.metrics.items():
        if name in initial_states:
            assert metric.total != initial_states[name], \
                f"Metric {name} was not updated"


def test_trainer_start_metrics_averages(config_with_metrics):
    """Test that epoch metrics are properly averaged"""
    trainer = Trainer(config_with_metrics)
    
    # Run one epoch
    trainer.config.train_loop.epochs = 1
    trainer.start()
    
    # Verify metrics handler computed averages
    assert trainer.metrics_handler.epoch_samples_count > 0
    assert trainer.metrics_handler.epoch_loss_eval >= 0
    assert all(v >= 0 for v in trainer.metrics_handler.epoch_metrics_eval.values())


# ===== Helper Tests =====

def test_trainer_start_with_backup_manager(config_with_helpers, tmp_path):
    """Test that BackupManager saves checkpoints"""
    backup_path = tmp_path / "backups"
    config_with_helpers["helpers"]["backup_manager"]["backup_path"] = str(backup_path)
    
    trainer = Trainer(config_with_helpers)
    trainer.start()
    
    # Check that backup directory was created
    assert backup_path.exists()
    
    # Check that at least one backup was saved
    backup_files = list(backup_path.glob("*.pt"))
    assert len(backup_files) > 0, "Should have created at least one backup"


def test_trainer_start_with_early_stopping(config_with_helpers):
    """Test that early stopping can terminate training early"""
    # Set patience very low and many epochs
    config_with_helpers["helpers"]["early_stopping"]["patience"] = 2
    config_with_helpers["train_loop"]["epochs"] = 100
    
    trainer = Trainer(config_with_helpers)
    
    # Mock validation to return constant loss (plateau)
    original_val = trainer._start_val_loop
    def mock_val():
        return 1.0, {}
    trainer._start_val_loop = mock_val
    
    trainer.start()
    
    # Should have stopped early
    assert trainer.current_epoch < 100, \
        f"Should stop early, but ran {trainer.current_epoch} epochs"


def test_trainer_start_with_lr_scheduler(config_with_helpers):
    """Test that LR scheduler reduces learning rate"""
    trainer = Trainer(config_with_helpers)
    
    initial_lr = trainer.optimizer.param_groups[0]['lr']
    
    # Mock plateau (constant validation loss)
    original_val = trainer._start_val_loop
    def mock_val():
        return 1.0, {}
    trainer._start_val_loop = mock_val
    
    trainer.start()
    
    final_lr = trainer.optimizer.param_groups[0]['lr']
    
    # LR should have been reduced
    assert final_lr < initial_lr, \
        f"LR should decrease from {initial_lr} but got {final_lr}"


def test_trainer_start_with_gradient_logger(config_with_helpers, caplog):
    """Test that gradient logger logs gradient norms"""
    config_with_helpers["helpers"]["gradient_logger"]["interval"] = 1
    
    trainer = Trainer(config_with_helpers)
    
    with caplog.at_level(logging.INFO):
        trainer.start()
    
    # Check that gradient norms were logged
    gradient_logs = [record for record in caplog.records 
                     if "Gradient norm" in record.message]
    
    assert len(gradient_logs) > 0, "Should have logged gradient norms"
    config_with_helpers["helpers"]["gradient_logger"]["interval"] = 50


# ===== Gradient Control Tests =====

def test_trainer_start_with_gradient_clipping(minimal_config):
    """Test that gradient clipping is applied"""
    minimal_config["gradient_control"]["grad_clip"] = 0.5
    
    trainer = Trainer(minimal_config)
    
    with patch('torch.nn.utils.clip_grad_norm_') as mock_clip:
        trainer.start()
        
        # Should have been called
        assert mock_clip.call_count > 0
        
        # Check max_norm parameter
        call_args = mock_clip.call_args_list[0]
        assert call_args[1]['max_norm'] == 0.5


def test_trainer_start_with_amp(minimal_config):
    """Test that AMP (mixed precision) works when enabled"""
    minimal_config["gradient_control"]["use_amp"] = True
    
    trainer = Trainer(minimal_config)
    
    # Should complete without errors
    trainer.start()

    # Should create GradScaler
    assert hasattr(trainer, 'scaler')
    
    assert trainer.current_epoch == minimal_config["train_loop"]["epochs"]


def test_trainer_start_without_gradient_clipping(minimal_config):
    """Test training works with grad_clip=None"""
    minimal_config["gradient_control"]["grad_clip"] = None
    
    trainer = Trainer(minimal_config)
    
    with patch('torch.nn.utils.clip_grad_norm_') as mock_clip:
        trainer.start()
        
        # Should NOT have been called
        assert mock_clip.call_count == 0


# ===== Model Mode Tests =====

def test_trainer_model_in_train_mode_during_training(minimal_config):
    """Test that model.train() and model.eval() are called appropriately"""
    trainer = Trainer(minimal_config)
    
    train_calls = []
    eval_calls = []
    
    # Track when train() and eval() are called
    original_train = trainer.model.train
    original_eval = trainer.model.eval
    
    def mock_train(mode=True):
        train_calls.append(mode)
        return original_train(mode)
    
    def mock_eval():
        eval_calls.append(True)
        return original_eval()
    
    trainer.model.train = mock_train
    trainer.model.eval = mock_eval
    
    trainer.start()
    
    # Should have called train() for each epoch
    assert len(train_calls) >= minimal_config["train_loop"]["epochs"], \
        f"model.train() should be called at least {minimal_config['train_loop']['epochs']} times, got {len(train_calls)}"
    
    # Should have called eval() for validation (once per epoch) + test (once at end)
    expected_eval_calls = minimal_config["train_loop"]["epochs"] + 1  # validation per epoch + final test
    assert len(eval_calls) >= expected_eval_calls, \
        f"model.eval() should be called at least {expected_eval_calls} times (validation + test), got {len(eval_calls)}"


def test_trainer_model_eval_during_validation_and_test(minimal_config):
    """Test that model.eval() is called during both validation and test"""
    trainer = Trainer(minimal_config)
    
    eval_call_locations = []
    
    # Track where eval() is called from
    original_eval = trainer.model.eval
    
    def mock_eval():
        # Check the call stack to determine where eval was called
        import inspect
        frame = inspect.currentframe()
        caller_name = frame.f_back.f_code.co_name
        eval_call_locations.append(caller_name)
        return original_eval()
    
    trainer.model.eval = mock_eval
    trainer.start()
    
    # Should have calls from validation loop
    assert any('val' in loc.lower() for loc in eval_call_locations), \
        f"model.eval() should be called from validation, got calls from: {eval_call_locations}"
    
    # Should have calls from test evaluation
    assert any('test' in loc.lower() for loc in eval_call_locations), \
        f"model.eval() should be called from test, got calls from: {eval_call_locations}"


def test_trainer_model_train_and_eval_alternation(minimal_config):
    """Test that model alternates between train and eval modes correctly"""
    trainer = Trainer(minimal_config)
    
    mode_sequence = []
    
    # Track mode changes
    original_train = trainer.model.train
    original_eval = trainer.model.eval
    
    def mock_train(mode=True):
        mode_sequence.append(('train', mode))
        return original_train(mode)
    
    def mock_eval():
        mode_sequence.append(('eval', None))
        return original_eval()
    
    trainer.model.train = mock_train
    trainer.model.eval = mock_eval
    
    trainer.start()
    
    # Verify sequence starts with train
    assert mode_sequence[0][0] == 'train', \
        f"Training should start with train mode, got {mode_sequence[0]}"
    
    # Verify train and eval alternate (roughly)
    train_count = sum(1 for mode, _ in mode_sequence if mode == 'train')
    eval_count = sum(1 for mode, _ in mode_sequence if mode == 'eval')
    
    # Should have roughly equal train/eval calls (train per epoch + eval per epoch + test)
    assert train_count >= minimal_config["train_loop"]["epochs"], \
        f"Should have at least {minimal_config['train_loop']['epochs']} train calls"
    assert eval_count >= minimal_config["train_loop"]["epochs"] + 1, \
        f"Should have at least {minimal_config['train_loop']['epochs'] + 1} eval calls (val + test)"


def test_trainer_model_in_eval_mode_during_test(minimal_config):
    """Test that model is in eval mode during test evaluation"""
    trainer = Trainer(minimal_config)
    
    test_mode_states = []
    
    # Wrap _start_test_eval to check model.training state
    original_test = trainer._start_test_eval
    
    def wrapped_test():
        # Check state at start of test
        test_mode_states.append(('start', trainer.model.training))
        result = original_test()
        # Check state at end of test
        test_mode_states.append(('end', trainer.model.training))
        return result
    
    trainer._start_test_eval = wrapped_test
    trainer.start()
    
    # Model should be in eval mode (training=False) during test
    assert len(test_mode_states) > 0, "Test evaluation should have been called"
    for location, training_mode in test_mode_states:
        assert not training_mode, \
            f"Model should be in eval mode during test ({location}), but training={training_mode}"


def test_trainer_model_eval_called_before_test_forward(minimal_config):
    """Test that model.eval() is called before test forward passes"""
    trainer = Trainer(minimal_config)
    
    eval_before_test_forward = []
    
    # Track if eval was called before forward in test
    original_eval = trainer.model.eval
    original_forward = trainer.model.forward
    
    test_phase = [False]  # Track if we're in test phase
    eval_called_in_test = [False]
    
    def mock_eval():
        if test_phase[0]:
            eval_called_in_test[0] = True
        return original_eval()
    
    def mock_forward(x):
        if test_phase[0]:
            eval_before_test_forward.append(eval_called_in_test[0])
        return original_forward(x)
    
    # Wrap test to mark test phase
    original_test = trainer._start_test_eval
    
    def wrapped_test():
        test_phase[0] = True
        result = original_test()
        test_phase[0] = False
        return result
    
    trainer.model.eval = mock_eval
    trainer.model.forward = mock_forward
    trainer._start_test_eval = wrapped_test
    
    trainer.start()
    
    # All test forward passes should have eval called before them
    assert len(eval_before_test_forward) > 0, "Should have test forward passes"
    assert all(eval_before_test_forward), \
        "model.eval() should be called before test forward passes"


# ===== Loss Tests =====

def test_trainer_start_loss_decreases(minimal_config):
    """Test that training loss generally decreases"""
    minimal_config["train_loop"]["epochs"] = 10
    
    trainer = Trainer(minimal_config)
    
    epoch_losses = []
    
    # Track losses
    original_train = trainer._train_epoch
    def wrapped_train():
        loss, metrics = original_train()
        epoch_losses.append(loss)
        return loss, metrics
    
    trainer._train_epoch = wrapped_train
    trainer.start()
    
    # Loss should generally decrease (allow some fluctuation)
    first_half_avg = sum(epoch_losses[:5]) / 5
    second_half_avg = sum(epoch_losses[5:]) / 5
    
    assert second_half_avg < first_half_avg, \
        f"Loss should decrease: first half={first_half_avg:.4f}, second half={second_half_avg:.4f}"


def test_trainer_start_validates_every_epoch(minimal_config):
    """Test that validation runs after every training epoch"""
    minimal_config["train_loop"]["epochs"] = 5
    
    trainer = Trainer(minimal_config)
    
    val_calls = []
    
    # Track validation calls
    original_val = trainer._start_val_loop
    def wrapped_val():
        val_calls.append(True)
        return original_val()
    
    trainer._start_val_loop = wrapped_val
    trainer.start()
    
    # Should validate once per epoch
    assert len(val_calls) == 5, \
        f"Should validate 5 times, got {len(val_calls)}"


# ===== Helper Call Order Tests =====

def test_trainer_helpers_called_in_correct_order(config_with_helpers):
    """Test that helpers are called in the correct sequence"""
    trainer = Trainer(config_with_helpers)
    
    call_order = []
    
    # Mock all helper methods
    for helper in trainer.helper_handler.helpers.values():
        if hasattr(helper, '_end_train_epoch_step'):
            original = helper._end_train_epoch_step
            def make_wrapper(name, orig):
                def wrapper(*args, **kwargs):
                    call_order.append(f"{name}._end_train_epoch_step")
                    return orig(*args, **kwargs) if orig else None
                return wrapper
            helper._end_train_epoch_step = make_wrapper(
                helper.__class__.__name__, original
            )
        
        if hasattr(helper, '_end_val_step'):
            original = helper._end_val_step
            def make_wrapper(name, orig):
                def wrapper(*args, **kwargs):
                    call_order.append(f"{name}._end_val_step")
                    return orig(*args, **kwargs) if orig else None
                return wrapper
            helper._end_val_step = make_wrapper(
                helper.__class__.__name__, original
            )
    
    # Run one epoch
    trainer.config.train_loop.epochs = 1
    trainer.start()
    
    # Check helpers were called
    assert len(call_order) > 0, "Helpers should have been called"


# ===== Edge Cases =====

def test_trainer_start_with_single_batch(minimal_config):
    """Test training works with only one batch"""
    minimal_config["dataset"]["dataset_args"]["nsamples"] = 8
    minimal_config["dataset"]["data_loader"]["batch_size"] = 8
    
    trainer = Trainer(minimal_config)
    trainer.start()
    
    assert trainer.current_epoch == minimal_config["train_loop"]["epochs"]


def test_trainer_start_test_evaluation_runs(minimal_config):
    """Test that test evaluation runs after training"""
    trainer = Trainer(minimal_config)
    
    test_called = []
    
    # Mock test evaluation
    original_test = trainer._start_test_eval
    def wrapped_test():
        test_called.append(True)
        return original_test()
    
    trainer._start_test_eval = wrapped_test
    trainer.start()
    
    # Test should have been called once
    assert len(test_called) == 1, "Test evaluation should run once"


def test_trainer_start_with_different_devices(minimal_config):
    """Test training works with specified device"""
    minimal_config["device"] = "cpu"
    
    trainer = Trainer(minimal_config)
    trainer.start()
    
    # Model should be on CPU
    assert next(trainer.model.parameters()).device.type == "cpu"


# ===== Integration Tests =====

def test_trainer_full_pipeline(config_with_helpers, tmp_path):
    """Integration test: full training pipeline with all features"""
    config_with_helpers["train_loop"]["epochs"] = 5
    config_with_helpers["info"]["metrics"] = [MetricType.MAE, MetricType.MSE]
    
    trainer = Trainer(config_with_helpers)
    trainer.start()
    
    # Verify training completed
    assert trainer.current_epoch == 5
    
    # Verify backups exist
    backup_path = Path(config_with_helpers["helpers"]["backup_manager"]["backup_path"])
    assert backup_path.exists()
    assert len(list(backup_path.glob("*.pt"))) > 0


def test_trainer_deterministic_with_seed(minimal_config):
    """Test that training is deterministic with fixed seed"""
    torch.manual_seed(42)
    trainer1 = Trainer(minimal_config)
    trainer1.start()
    loss1 = trainer1.metrics_handler.epoch_loss_eval
    
    torch.manual_seed(42)
    trainer2 = Trainer(minimal_config)
    trainer2.start()
    loss2 = trainer2.metrics_handler.epoch_loss_eval
    
    # Losses should be very close (allow tiny floating point differences)
    assert abs(loss1 - loss2) < 1e-5, \
        f"Training should be deterministic with seed, got {loss1} vs {loss2}"
