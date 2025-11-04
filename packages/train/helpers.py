import logging
import os
from re import match
from typing import Callable, Dict

import matplotlib.pyplot as plt
import torch

from torch.optim.lr_scheduler import ReduceLROnPlateau
from packages.plotting import train_plots
from packages.train.metrics import TorchMetric
from packages.train.trainer_config_schema import HistoryPlot, PlotStates
from packages.train.loss import TorchLoss
from packages.train.trainer_config_schema import TrainerConfig
from typing import List

class Helper():
    """
    Base class for all helpers.
    Helpers are implemented by implementing at least one of the following methods:
    - _end_train_epoch_step(self, **kwargs)
    - _end_val_step(self, **kwargs)
    These methods will be called at the end of each training epoch and validation step respectively.
    kwargs can contain any relevant information such as epoch number, loss values, metrics, etc.
    """
    def __init__(self):
        pass

class BackupManager(Helper):
    def __init__(self, backup_interval, backup_path, model):
        self.backup_interval = backup_interval
        self.backup_path = backup_path
        self.last_backup_epoch = -1
        os.makedirs(backup_path, exist_ok=True)
        self.best_loss = None

        self.model = model

        self.best_model_backup_file = ""

    def _end_val_step(self, **kwargs):
        epoch = kwargs.get("epoch")
        val_loss = kwargs.get("val_loss")
        self._best_model_backup(val_loss, epoch)
        self._periodic_backup(epoch)

    def _best_model_backup(self, loss, epoch):
        if self.best_loss is None or loss < self.best_loss:
            backup_file = os.path.join(
                self.backup_path, f"best_model_epoch_{epoch + 1}.pt"
            )
            torch.save(self.model.state_dict(), backup_file)
            logging.info(
                f"Best model saved at epoch {epoch + 1} with loss {loss:.4f}"
            )
            self.best_model_backup_file = backup_file
            self.best_loss = loss
            self.last_backup_epoch = epoch

    def _periodic_backup(self, epoch):
        if (epoch + 1) % self.backup_interval == 0 and epoch != self.last_backup_epoch:
            backup_file = os.path.join(self.backup_path, f"model_epoch_{epoch + 1}.pt")
            torch.save(self.model.state_dict(), backup_file)
            logging.info(f"Periodic backup saved at epoch {epoch + 1}")
            self.last_backup_epoch = epoch

class EarlyStopping(Helper):
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.stop_early = False


    def _end_val_step(self, **kwargs):
        assert 'val_loss' in kwargs, "val_loss must be provided to EarlyStopping helper."
        loss = kwargs.get("val_loss")
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop_early = True
                logging.info("Early stopping triggered.")


class History(Helper):
    def __init__(
        self,
        save_path: str,
        plot_type: PlotStates,
        metrics: List[TorchMetric],
        loss: TorchLoss
    ):
        self.save_path = save_path
        self.plot_type = plot_type
        
        self._init_histories(metrics, loss)

    def _init_histories(self, metrics: List[TorchMetric], loss: TorchLoss):
        self.metrics = [metric.name for metric in metrics]
        self.train_history = {loss.name: []}
        self.val_history = {loss.name: []}
        for metric_name in self.metrics:
            self.train_history[metric_name] = []
            self.val_history[metric_name] = []

    def _end_train_epoch_step(self, **kwargs):
        assert 'train_loss' in kwargs, "train_loss must be provided to History helper."
        train_loss = kwargs.get("train_loss")
        self.train_history, log_string = self._log_history(
            self.train_history, kwargs.get("metrics"), train_loss
        )

        logging.info("Training" + log_string)

    def _end_val_step(self, **kwargs):
        val_loss = kwargs.get("val_loss")
        metrics_eval = kwargs.get("metrics_eval")

        self.val_history, log_string = self._log_history(
            self.val_history, metrics_eval, {"val_" + k: v for k, v in val_loss.items()}
        )

        logging.info("Validation" + log_string)

    @staticmethod
    def _log_history(curr_history: Dict[str, List[float]], metrics: Dict[str, float], loss: Dict[str, float]):

        log_string = f"_Loss:"
        for loss_key, loss_value in loss.items():
            curr_history[loss_key].append(loss_value)
            log_string += f", {loss_key}: {loss_value:.4f}"

        log_string += f"\n Metrics:"
        for metric_name, metric_value in metrics.items():
            curr_history[metric_name].append(metric_value)
            log_string += f", {metric_name}: {metric_value:.4f}"

        return curr_history, log_string

    def plot_history(self):
        match(self.plot_type):
            case PlotStates.Extended:
                train_plots.plot_history_extended(self)
                self.save_plot()
            case PlotStates.Tight:
                train_plots.plot_history_tight(self)
                self.save_plot()
            case PlotStates.Off:
                return
            case _:
                raise ValueError(
                    "Invalid value for 'plot_type'. Choose from 'extended', 'tight', or None."
                )

    def save_plot(self):
        if self.save_path:
            plt.savefig(self.save_path)
            logging.info(f"Training history plot saved to {self.save_path}")

# TODO Do both batch and epoch tracking and setup a better visualization
class GradientLogger(Helper):
    def __init__(self, model, interval= None):
        self.interval = interval
        self.counter = 0
        self.model = model

    def _end_train_epoch_step(self, **kwargs):
        if self.interval is None:
            return
        self.counter += 1
        if self.counter % self.interval == 0:
            for name, param in self.model.named_parameters():
                logging.info(f"Gradient norm for {name}: {param.grad.norm().item():.4f}")


class LRScheduler(Helper):
    def __init__(self, scheduler: ReduceLROnPlateau):
        self.scheduler = scheduler
        self._last_lr = None

    def _end_val_step(self, **kwargs):
        val_loss = kwargs.get("val_loss")

        self.scheduler.step(val_loss)
        if self._last_lr != self.scheduler.get_last_lr()[0]:
            self._last_lr = self.scheduler.get_last_lr()[0]
            logging.info(f"Updated Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")




class HelperHandler:
    def __init__(self, config: TrainerConfig, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self._initialize_helpers()
        
    def _call_end_train_epoch_step(self, epoch, train_loss, metrics):
        input = {'epoch': epoch, 'train_loss': train_loss, 'metrics': metrics}
        for helper in self.helpers:
            if hasattr(helper, '_end_train_epoch_step'): 
                helper._end_train_epoch_step(**input)

    def _call_end_val_step(self, epoch, val_loss, val_metrics):
        input = {'epoch': epoch, 'val_loss': val_loss, 'metrics': val_metrics}
        for helper in self.helpers:
            if hasattr(helper, '_end_val_step'):
                helper._end_val_step(**input)

    def _initialize_helpers(self):
        selected_helpers = self.config.helpers.model_dump().keys()

        self.helpers = []
        for helper_name in selected_helpers:
            match helper_name:
                case "history":
                    self.history_manager = History(
                    save_path=self.config.info.history_plot.save_path,
                    plot_type=self.config.info.history_plot.plot_type,
                    loss=self.config.get_loss_class(),
                    metrics=self.config.info.get_metric_classes()
                )
                case "reduce_lr_on_plateau":
                    helper_instance = LRScheduler(
                        scheduler= torch.optim.lr_scheduler.ReduceLROnPlateau(
                            self.optimizer,
                            **self.config.helpers.reduce_lr_on_plateau.model_dump()
                        )
                    )
                case "gradient_logger":
                    helper_instance = GradientLogger(
                        model=self.model,
                        interval=self.config.helpers.gradient_logger.interval
                    )
                case "backup_manager":
                    helper_instance = BackupManager(
                        backup_interval=self.config.helpers.backup_manager.backup_interval,
                        backup_path=self.config.helpers.backup_manager.backup_path,
                        model=self.model
                    )
                case "early_stopping":
                    helper_instance = EarlyStopping(
                        patience=self.config.helpers.early_stopping.patience,
                        min_delta=self.config.helpers.early_stopping.min_delta
                    )
                    self.stop_early = helper_instance.stop_early
                case _:
                    logging.warning(f"Unknown helper: {helper_name}, skipping initialization.")
                    continue
            self.helpers.append(helper_instance)
